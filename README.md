# GSHS 2025 EDGE Research

**GPT가 작성한 프로젝트에 대한 설명입니다**

### 환경 설정하기

1. **UV 설치**: 아직 설치되어 있지 않다면 아래 명령어를 사용하세요.
   * Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   * Windows:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
2. **Python 패키지 설치**:
   ```bash
   uv python install
   ```
3. **환경 동기화**:
   ```bash
   uv sync
   ```
4. **학습 실행**:
   ```bash
   uv run python src/train.py
   ```
### 설정(Configuration)
* 메인 설정 파일은 `configs/config.yaml`에 위치해 있습니다. 실험에 사용할 파라미터를 설정하려면 이 파일을 수정하세요.
### 새로운 모델 또는 학습 로직 구현하기
* 새로운 모델이나 학습 로직을 구현하려면 `src/models/` 디렉토리에 새로운 클래스를 생성하세요. 일관성을 유지하기 위해 기존 구조를 따르도록 하세요.
### 헬퍼 클래스 사용하기
* `src/logging/train_logger.py`에 있는 `TrainLogger` 클래스는 학습 메트릭을 로깅하는 데 사용됩니다. 로그를 저장할 디렉토리를 지정하여 초기화하세요:
  ```python
  logger = TrainLogger(run_dir="로그를 저장할/경로")
  ```
* `src/logging/log_drawer.py`에 있는 `LogDrawer` 클래스는 JSONL 형식으로 저장된 학습 메트릭을 시각화하는 데 사용할 수 있습니다. 동일하게 초기화하세요:
  ```python
  drawer = LogDrawer(run_dir="로그가 있는/경로")
  ```
### 추가 참고 사항
* 모든 의존성은 `pyproject.toml`에 명시된 대로 설치되어야 합니다. 환경 관리는 `uv sync`를 사용하세요.
