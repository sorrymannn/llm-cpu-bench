#!/usr/bin/env python3
"""
llm-cpu-bench — AI 워크로드에서 CPU 캐시 효과 벤치마크
======================================================
GPU가 LLM 추론(Prefill/Decode)을 하는 동안,
CPU가 담당하는 작업들에서 L3 캐시 크기가 성능에 미치는 영향을 측정합니다.

측정 항목 (슬라이드 "What tasks CPU is doing for Local LLM Inference" 기반):
  1. Cache Latency Profile — L1/L2/L3/DRAM 경계에서의 접근 속도 변화
  2. Tokenizer Vocab Lookup — Tokenize/Detokenize 시 해시 테이블 랜덤 룩업
  3. RAG Vector Search     — RAG Context Retrieval 시 임베딩 벡터 유사도 검색

외부 도구 없이 순수 Python만으로 실행됩니다.
numpy가 설치되어 있으면 더 빠르게 돌아갑니다 (없어도 동작).

사용법:
  python benchmark.py --output 9800x3d.json
  python benchmark.py --output 9700x.json --label "Ryzen 7 9700X"
"""

import argparse
import array
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─────────────────────────────────────────────
# 시스템 정보
# ─────────────────────────────────────────────

def get_cpu_info():
    phys, logical = _detect_cores()
    info = {
        "model": "Unknown", "cores_physical": phys, "cores_logical": logical,
        "l3_cache": "Unknown", "architecture": platform.machine(),
    }
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "Model name" in line:
                    info["model"] = line.split(":", 1)[1].strip()
                elif "L3 cache" in line:
                    info["l3_cache"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            info["model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
        try:
            l3 = int(subprocess.check_output(["sysctl", "-n", "hw.l3cachesize"], text=True).strip())
            info["l3_cache"] = f"{l3 // 1024 // 1024} MiB"
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name,L3CacheSize", "/format:list"],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "Name":
                    info["model"] = v.strip()
                elif k.strip() == "L3CacheSize" and v.strip():
                    val = int(v.strip())
                    if val > 0:
                        info["l3_cache"] = f"{val // 1024} MiB"
        except Exception:
            pass
    return info


def _detect_cores():
    try:
        import multiprocessing
        logical = multiprocessing.cpu_count()
    except Exception:
        logical = os.cpu_count() or 4
    physical = max(1, logical // 2)
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            c, s = 1, 1
            for line in out.splitlines():
                if line.strip().startswith("Core(s) per socket"):
                    c = int(line.split(":")[1].strip())
                elif line.strip().startswith("Socket(s)"):
                    s = int(line.split(":")[1].strip())
            physical = c * s
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            physical = int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores", "/format:list"],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                if "NumberOfCores=" in line:
                    physical = int(line.split("=")[1].strip())
                    break
        except Exception:
            pass
    return max(1, physical), logical


def get_memory_info():
    info = {"total_gb": 0, "speed": "Unknown"}
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "memorychip", "get", "Capacity,ConfiguredClockSpeed", "/format:list"],
                text=True, stderr=subprocess.DEVNULL)
            caps, speeds = [], []
            for line in out.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "Capacity" and v.strip():
                    caps.append(int(v.strip()))
                elif k.strip() == "ConfiguredClockSpeed" and v.strip():
                    speeds.append(int(v.strip()))
            if caps:
                info["total_gb"] = round(sum(caps) / 1024**3, 1)
            if speeds:
                info["speed"] = f"{max(speeds)} MT/s"
        except Exception:
            pass
    return info


def get_os_info():
    info = {"system": platform.system(), "release": platform.release()}
    if platform.system() == "Linux":
        try:
            info["kernel"] = subprocess.check_output(["uname", "-r"], text=True).strip()
        except Exception:
            pass
    return info


# ─────────────────────────────────────────────
# 편차 제어
# ─────────────────────────────────────────────

def apply_variance_controls():
    controls = []
    if platform.system() == "Linux":
        for name, cmd in [
            ("CPU governor -> performance",
             ["sudo", "bash", "-c",
              "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done"]),
            ("NUMA balancing -> disabled",
             ["sudo", "sysctl", "-w", "kernel.numa_balancing=0"]),
            ("THP -> disabled",
             ["sudo", "bash", "-c", "echo never > /sys/kernel/mm/transparent_hugepage/enabled"]),
        ]:
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                controls.append(name)
            except Exception:
                controls.append(f"{name} (SKIPPED)")
    elif platform.system() == "Windows":
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00000080)
            controls.append("Process priority -> HIGH")
        except Exception:
            controls.append("Process priority -> default")
    return controls


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def measure(func, warmup=2, runs=5):
    """함수를 warmup + runs번 실행하고 통계 반환"""
    # 워밍업
    for _ in range(warmup):
        func()
    time.sleep(0.5)

    # 본 측정
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        time.sleep(0.2)

    mean_t = statistics.mean(times)
    stdev_t = statistics.stdev(times) if len(times) > 1 else 0
    cv = (stdev_t / mean_t * 100) if mean_t > 0 else 0

    return {
        "times": [round(t, 6) for t in times],
        "mean_sec": round(mean_t, 6),
        "median_sec": round(statistics.median(times), 6),
        "stdev_sec": round(stdev_t, 6),
        "cv_percent": round(cv, 2),
        "min_sec": round(min(times), 6),
        "max_sec": round(max(times), 6),
    }


# ─────────────────────────────────────────────
# TEST 1: Cache Latency Profile
# ─────────────────────────────────────────────
# 다양한 크기의 배열에서 랜덤 포인터 체이싱 수행.
# 배열이 L3 캐시 안에 들어가면 빠르고, 넘어가면 DRAM 속도로 떨어짐.
# X3D(96MB L3): ~96MB까지 빠름
# Non-X3D(32MB L3): ~32MB에서 이미 느려짐

def _make_chase_array(n_elements):
    """랜덤 포인터 체이스 배열 생성 (Fisher-Yates shuffle)"""
    indices = list(range(n_elements))
    # 순환 랜덤 순열 생성
    for i in range(n_elements - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    # 순환 체인으로 변환
    arr = array.array('i', [0] * n_elements)
    for i in range(n_elements):
        arr[indices[i]] = indices[(i + 1) % n_elements]
    return arr


def _chase_pointers(arr, steps):
    """포인터 체이싱 — 캐시 레이턴시 측정의 핵심"""
    idx = 0
    for _ in range(steps):
        idx = arr[idx]
    return idx


def test_cache_latency(runs=5, verbose=True):
    """캐시 레이턴시 프로파일 측정"""
    if verbose:
        print("\n" + "=" * 60)
        print("[Test 1] Cache Latency Profile")
        print("  Random pointer chasing across different buffer sizes")
        print("  X3D stays fast up to ~96MB, Non-X3D drops at ~32MB")
        print("=" * 60)

    # 테스트할 버퍼 크기 (KB)
    # 4 bytes per int element
    sizes_kb = [
        16, 32, 64,                  # L1 (~32KB)
        128, 256, 512, 1024,         # L2 (~1MB)
        2048, 4096, 8192,            # L3 시작
        16384, 32768,                # 32MB — Non-X3D L3 경계
        49152, 65536,                # Non-X3D는 DRAM, X3D는 아직 L3
        98304, 131072,               # 96MB — X3D L3 경계
        196608,                      # 둘 다 DRAM
    ]

    chase_steps = 2_000_000  # 충분한 반복으로 시간 측정 정확도 확보
    results = []

    for size_kb in sizes_kb:
        n_elements = (size_kb * 1024) // 4  # 4 bytes per int
        size_mb = size_kb / 1024

        if verbose:
            print(f"\n  {size_mb:>8.1f} MB ... ", end="", flush=True)

        try:
            arr = _make_chase_array(n_elements)
        except MemoryError:
            if verbose:
                print("SKIP (not enough memory)")
            continue

        def run_chase():
            return _chase_pointers(arr, chase_steps)

        stats = measure(run_chase, warmup=1, runs=runs)
        ns_per_access = (stats["mean_sec"] / chase_steps) * 1e9

        entry = {
            "size_kb": size_kb,
            "size_mb": round(size_mb, 1),
            "ns_per_access": round(ns_per_access, 2),
            "total_sec": stats["mean_sec"],
            "cv_percent": stats["cv_percent"],
        }
        results.append(entry)

        if verbose:
            quality = "OK" if stats["cv_percent"] <= 3 else "NOISY"
            print(f"{ns_per_access:>8.2f} ns/access  (CV: {stats['cv_percent']:.1f}% {quality})")

        del arr

    return results


# ─────────────────────────────────────────────
# TEST 2: Tokenizer Vocab Lookup
# ─────────────────────────────────────────────
# LLM Tokenize/Detokenize에서 CPU가 하는 핵심 작업:
# 수만 개의 토큰 어휘(vocab) 테이블에서 랜덤 해시 룩업.
# vocab 테이블이 L3 캐시에 들어가면 빠름.
# X3D: 큰 vocab도 캐시에 유지 가능.

def _build_vocab_table(vocab_size, token_length=12):
    """토크나이저 어휘 테이블 시뮬레이션 (dict 기반 해시 테이블)"""
    vocab = {}
    for i in range(vocab_size):
        # 실제 토크나이저처럼 바이트 시퀀스를 키로 사용
        key = hashlib.md5(str(i).encode()).hexdigest()[:token_length]
        vocab[key] = i
    return vocab, list(vocab.keys())


def test_tokenizer_lookup(runs=5, verbose=True):
    """토크나이저 Vocab 랜덤 룩업 벤치마크"""
    if verbose:
        print("\n" + "=" * 60)
        print("[Test 2] Tokenizer Vocab Lookup")
        print("  Simulates Tokenize/Detokenize random hash table access")
        print("  Larger vocab + larger cache = faster lookup")
        print("=" * 60)

    # 다양한 vocab 크기 테스트
    # 실제 LLM: LLaMA=32K, GPT-4=100K, 한국어 모델=64K+
    vocab_configs = [
        {"size": 32_000,  "name": "32K (LLaMA 2)",     "lookups": 500_000},
        {"size": 64_000,  "name": "64K (multilingual)", "lookups": 500_000},
        {"size": 128_000, "name": "128K (GPT-4 class)", "lookups": 500_000},
        {"size": 256_000, "name": "256K (large vocab)",  "lookups": 300_000},
    ]

    results = []

    for config in vocab_configs:
        if verbose:
            print(f"\n  Vocab: {config['name']} ({config['lookups']:,} lookups) ... ", end="", flush=True)

        vocab, keys = _build_vocab_table(config["size"])
        n_keys = len(keys)
        n_lookups = config["lookups"]

        # 랜덤 룩업 순서 미리 생성
        lookup_indices = [random.randint(0, n_keys - 1) for _ in range(n_lookups)]

        def run_lookup():
            total = 0
            for idx in lookup_indices:
                total += vocab[keys[idx]]
            return total

        stats = measure(run_lookup, warmup=2, runs=runs)
        lookups_per_sec = n_lookups / stats["mean_sec"]

        entry = {
            "vocab_size": config["size"],
            "vocab_name": config["name"],
            "num_lookups": n_lookups,
            "lookups_per_sec": round(lookups_per_sec),
            "mean_sec": stats["mean_sec"],
            "cv_percent": stats["cv_percent"],
        }
        results.append(entry)

        if verbose:
            print(f"{lookups_per_sec:>12,.0f} lookups/sec  (CV: {stats['cv_percent']:.1f}%)")

        del vocab, keys

    return results


# ─────────────────────────────────────────────
# TEST 3: RAG Vector Search Simulation
# ─────────────────────────────────────────────
# RAG Context Retrieval에서 CPU가 하는 핵심 작업:
# 임베딩 벡터 DB에서 쿼리 벡터와 유사한 벡터를 찾기 위해
# 랜덤한 벡터들에 대해 코사인 유사도를 계산.
# 벡터 DB가 L3에 들어가면 검색이 훨씬 빠름.

def _cosine_similarity_pure(a, b):
    """순수 Python 코사인 유사도"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_similarity_np(a, b):
    """numpy 코사인 유사도"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_rag_vector_search(runs=5, verbose=True):
    """RAG 벡터 검색 시뮬레이션 벤치마크"""
    if verbose:
        print("\n" + "=" * 60)
        print("[Test 3] RAG Vector Search Simulation")
        print("  Simulates embedding similarity search (cosine distance)")
        print("  Random access pattern — heavily cache-dependent")
        print("=" * 60)

    # 벡터 DB 설정
    # 실제 RAG: 768~1536 차원, 수천~수십만 문서
    configs = [
        {"n_vectors": 10_000,  "dim": 768,  "queries": 100, "name": "10K docs, 768d"},
        {"n_vectors": 50_000,  "dim": 768,  "queries": 50,  "name": "50K docs, 768d"},
        {"n_vectors": 10_000,  "dim": 1536, "queries": 100, "name": "10K docs, 1536d"},
        {"n_vectors": 100_000, "dim": 384,  "queries": 50,  "name": "100K docs, 384d"},
    ]

    results = []

    for config in configs:
        n_vec = config["n_vectors"]
        dim = config["dim"]
        n_queries = config["queries"]
        db_size_mb = (n_vec * dim * 4) / (1024 * 1024)  # float32

        if verbose:
            print(f"\n  {config['name']} (DB: {db_size_mb:.0f} MB, {n_queries} queries) ... ",
                  end="", flush=True)

        # 벡터 DB 생성
        if HAS_NUMPY:
            db = np.random.randn(n_vec, dim).astype(np.float32)
            # 정규화
            norms = np.linalg.norm(db, axis=1, keepdims=True)
            norms[norms == 0] = 1
            db = db / norms
            queries = np.random.randn(n_queries, dim).astype(np.float32)
            q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
            q_norms[q_norms == 0] = 1
            queries = queries / q_norms

            def run_search():
                """각 쿼리에 대해 brute-force 유사도 계산 후 top-5 반환"""
                top_k = 5
                all_results = []
                for q_idx in range(n_queries):
                    # 실제 RAG처럼 쿼리별로 전체 DB 스캔 (또는 랜덤 접근)
                    scores = db @ queries[q_idx]
                    top_indices = np.argpartition(scores, -top_k)[-top_k:]
                    all_results.append(top_indices)
                return all_results
        else:
            # 순수 Python fallback
            db = []
            for _ in range(n_vec):
                vec = [random.gauss(0, 1) for _ in range(dim)]
                norm = math.sqrt(sum(x*x for x in vec))
                if norm > 0:
                    vec = [x/norm for x in vec]
                db.append(vec)
            queries_list = []
            for _ in range(n_queries):
                vec = [random.gauss(0, 1) for _ in range(dim)]
                norm = math.sqrt(sum(x*x for x in vec))
                if norm > 0:
                    vec = [x/norm for x in vec]
                queries_list.append(vec)

            def run_search():
                top_k = 5
                all_results = []
                for q in queries_list:
                    scores = []
                    for i, doc in enumerate(db):
                        s = _cosine_similarity_pure(q, doc)
                        scores.append((s, i))
                    scores.sort(reverse=True)
                    all_results.append([idx for _, idx in scores[:top_k]])
                return all_results

        stats = measure(run_search, warmup=1, runs=runs)
        queries_per_sec = n_queries / stats["mean_sec"]

        entry = {
            "n_vectors": n_vec,
            "dimension": dim,
            "n_queries": n_queries,
            "db_size_mb": round(db_size_mb, 1),
            "queries_per_sec": round(queries_per_sec, 2),
            "mean_sec": stats["mean_sec"],
            "cv_percent": stats["cv_percent"],
            "backend": "numpy" if HAS_NUMPY else "pure_python",
        }
        results.append(entry)

        if verbose:
            print(f"{queries_per_sec:>8.1f} queries/sec  (CV: {stats['cv_percent']:.1f}%)")

        del db

    return results


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="llm-cpu-bench: AI 워크로드에서 CPU 캐시 효과 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python benchmark.py --output 9800x3d.json
  python benchmark.py --output 9700x.json --label "Ryzen 7 9700X"
  python benchmark.py --output my_cpu.json --runs 10

비교:
  python visualize.py 9800x3d.json 9700x.json
        """
    )
    parser.add_argument("--output", "-o", required=True, help="결과 JSON 파일 경로")
    parser.add_argument("--label", default=None, help="결과 레이블 (기본: CPU 이름 자동 감지)")
    parser.add_argument("--runs", "-r", type=int, default=5, help="테스트당 반복 횟수 (기본: 5)")
    parser.add_argument("--skip-controls", action="store_true", help="환경 제어 건너뛰기")

    args = parser.parse_args()

    # 시스템 정보
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()
    os_info = get_os_info()
    label = args.label or cpu_info["model"]

    print("=" * 60)
    print("llm-cpu-bench — CPU Cache Benchmark for AI Workloads")
    print("=" * 60)
    print(f"\n  CPU:      {cpu_info['model']}")
    print(f"  Cores:    {cpu_info['cores_physical']}P / {cpu_info['cores_logical']}T")
    print(f"  L3 Cache: {cpu_info['l3_cache']}")
    print(f"  Memory:   {mem_info['total_gb']} GB @ {mem_info['speed']}")
    print(f"  OS:       {os_info['system']} {os_info['release']}")
    print(f"  numpy:    {'Yes' if HAS_NUMPY else 'No (pure Python fallback)'}")
    print(f"  Label:    {label}")
    print(f"  Runs:     {args.runs}")

    # 편차 제어
    if not args.skip_controls:
        controls = apply_variance_controls()
        if controls:
            print(f"\n  Variance controls:")
            for c in controls:
                print(f"    * {c}")
    else:
        controls = ["SKIPPED"]

    # 테스트 실행
    t0 = time.time()

    cache_results = test_cache_latency(runs=args.runs)
    tokenizer_results = test_tokenizer_lookup(runs=args.runs)
    rag_results = test_rag_vector_search(runs=args.runs)

    elapsed = time.time() - t0

    # 결과 저장
    output = {
        "meta": {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "benchmark_version": "2.0.0",
            "tool": "llm-cpu-bench",
            "url": "https://github.com/sorrymannn/llm-cpu-bench",
        },
        "system": {
            "cpu": cpu_info,
            "memory": mem_info,
            "os": os_info,
            "numpy": HAS_NUMPY,
        },
        "config": {
            "runs": args.runs,
            "variance_controls": controls,
        },
        "results": {
            "cache_latency": cache_results,
            "tokenizer_lookup": tokenizer_results,
            "rag_vector_search": rag_results,
        },
    }

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # 요약
    print(f"\n{'=' * 60}")
    print(f"Done! ({elapsed:.0f}s)")
    print(f"Saved: {args.output}")
    print(f"{'=' * 60}")

    print(f"\n--- Cache Latency ---")
    print(f"  {'Size':>10}  {'ns/access':>10}")
    for r in cache_results:
        marker = ""
        if r["size_mb"] == 32:
            marker = "  <-- Non-X3D L3 boundary"
        elif r["size_mb"] == 96:
            marker = "  <-- X3D L3 boundary"
        print(f"  {r['size_mb']:>8.1f} MB  {r['ns_per_access']:>10.2f}{marker}")

    print(f"\n--- Tokenizer Lookup ---")
    for r in tokenizer_results:
        print(f"  {r['vocab_name']:<24} {r['lookups_per_sec']:>12,} lookups/sec")

    print(f"\n--- RAG Vector Search ---")
    for r in rag_results:
        print(f"  {r['n_vectors']:>6} vecs x {r['dimension']}d  ({r['db_size_mb']:>5.0f} MB)  "
              f"{r['queries_per_sec']:>8.1f} queries/sec")

    print(f"\nNext: python visualize.py {args.output} <other_result>.json")


if __name__ == "__main__":
    main()
