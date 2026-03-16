# llm-cpu-bench

[![License: MIT](https://img.shields.io/github/license/sorrymannn/llm-cpu-bench)](https://github.com/sorrymannn/llm-cpu-bench)

CPU 캐시가 LLM 추론에 미치는 영향을 측정하는 벤치마크

**Repository:** https://github.com/sorrymannn/llm-cpu-bench

## 이게 뭐하는 건가요?

AI PC에서 LLM을 돌릴 때, GPU만 일하는 게 아닙니다.
GPU가 Prefill/Decode를 하는 동안 **CPU가 담당하는 작업**들이 있습니다:

| CPU가 하는 일 | 특성 |
|---|---|
| **Tokenize / Detokenize** | 어휘 테이블 랜덤 룩업 |
| **RAG Vector Searching** | 임베딩 벡터 유사도 계산 |
| **GPU data feeding** | KV 캐시 관리, 배치 준비 |

이 작업들의 공통점: **Latency-Sensitive**, **Random Memory Access Pattern**

이 두 특성은 정확히 **L3 캐시 크기에 민감한 워크로드**입니다.
X3D(96MB L3)와 Non-X3D(32MB L3)에서 이 작업들의 속도가 실제로 얼마나 다른지를 측정합니다.

## 뭘 측정하나요?

| 테스트 | 대응하는 실제 작업 | 측정 방법 |
|---|---|---|
| **Cache Latency Profile** | 전체 파이프라인 | 버퍼 크기별 랜덤 접근 레이턴시 → L3 경계에서 속도 변화 |
| **Tokenizer Vocab Lookup** | Tokenize / Detokenize | 다양한 vocab 크기의 해시 테이블 랜덤 룩업 속도 |
| **RAG Vector Search** | RAG Context Retrieval | 임베딩 벡터 DB에서 코사인 유사도 brute-force 검색 |

### 왜 X3D가 유리한가?

```
Non-X3D (9700X):  L3 32MB → 32MB 넘는 데이터 접근 시 DRAM으로 (~70ns)
X3D (9800X3D):    L3 96MB → 96MB까지 L3 캐시 히트 (~4-5ns)

레이턴시 차이: 약 15배
```

벤치마크의 핵심 차트인 **Cache Latency Profile**에서 이 경계를 직접 눈으로 확인할 수 있습니다.

## 준비물

- **Python 3.8+**
- **numpy** (권장, 없어도 동작)

```bash
pip install numpy matplotlib
```

> matplotlib는 시각화(visualize.py)에만 필요합니다.
> benchmark.py 실행에는 Python만 있으면 됩니다.

## 설치 및 실행

### 1단계: 이 저장소 받기

```bash
git clone https://github.com/sorrymannn/llm-cpu-bench.git
cd llm-cpu-bench
```

### 2단계: 벤치마크 실행

**Windows:**
```powershell
python benchmark.py --output my_cpu.json
```

**Linux / macOS:**
```bash
python3 benchmark.py --output my_cpu.json
```

커스텀 레이블:
```bash
python benchmark.py --output 9800x3d.json --label "Ryzen 7 9800X3D"
```

> 외부 도구(llama.cpp 등) 설치 필요 없음. 순수 Python으로 동작합니다.

### 3단계: 다른 CPU에서도 실행

비교할 CPU 시스템에서 동일하게 실행합니다.

```bash
python benchmark.py --output 9700x.json --label "Ryzen 7 9700X"
```

### 4단계: 결과 비교

```powershell
# Windows
python visualize.py 9800x3d.json 9700x.json

# Linux / macOS
python3 visualize.py 9800x3d.json 9700x.json
```

2개 이상도 가능:
```bash
python3 visualize.py 9800x3d.json 9700x.json i7_14700k.json
```

### 생성되는 차트

| 파일 | 설명 |
|---|---|
| `01_cache_latency_profile.png` | **핵심 차트** — 버퍼 크기별 레이턴시 (32MB/96MB 경계 표시) |
| `02_tokenizer_lookup.png` | Vocab 크기별 토크나이저 룩업 속도 |
| `03_rag_vector_search.png` | 벡터 DB 크기별 RAG 검색 속도 |
| `04_summary.png` | 전체 결과 요약 (% 차이) |

## 편차 제어

자동으로 적용됩니다 (권한 없으면 스킵):

- CPU governor performance 고정 (Linux)
- NUMA balancing 비활성화 (Linux)
- THP 비활성화 (Linux)
- 프로세스 우선순위 상향 (Windows/Linux/macOS)
- 워밍업 2회 + 본 측정 5회 + 런 간 대기

## benchmark.py 옵션

```
python benchmark.py --output <파일.json> [옵션]

  --output, -o     결과 JSON 파일 경로 (필수)
  --label          결과 레이블 (기본: CPU 이름 자동 감지)
  --runs, -r       테스트당 반복 횟수 (기본: 5)
  --skip-controls  환경 제어 건너뛰기
```

## BIOS 설정 권장

정확한 비교를 위해 양쪽 시스템에서 동일하게 설정:

- **PBO / Turbo Boost**: 양쪽 동일 (둘 다 켜거나 둘 다 끄기)
- **XMP / EXPO**: 같은 메모리 프로파일 (예: DDR5-6000 CL30)
- **C-States**: 끄기
- **Cool & Quiet**: 끄기

## 기여 방법

1. 자기 CPU에서 벤치마크 실행
2. 결과 JSON과 함께 Issue 또는 PR 열기
3. 포함할 정보: CPU 모델, 메인보드, BIOS 버전, 메모리 설정, OS

## License

MIT
