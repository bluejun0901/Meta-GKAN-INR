# GSHS 2025 EDGE Research

### 실행하는 방법
```
# uv 설치 (없으면)(맥 / 리눅스)
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv 설치 (없으면)(윈도우)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Python 맞추기
uv python install

# 환경 복원
uv sync

# 실행
uv run python src/train.py
```