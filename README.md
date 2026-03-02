# GSHS 2025 EDGE Research

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

2. **가상환경 생성 및 의존성 다운로드**:
    프로젝트 폴더에서
    ```bash
    uv venv
    uv sync
    ```
    를 실행하세요. venv가상환경이 생성되며, torch등의 페키지가 다운로드됩니다.

3. **데이터셋 다운로드**:
    dataset폴더를 만든 이후, 다음 명령어를 실행하면 STI데이터셋이 다운로드됩니다.
    ```bash
    curl -L -o ./dataset/standard-test-images.zip\
    https://www.kaggle.com/api/v1/datasets/download/saeedehkamjoo/standard-test-images
    ```
    다운로드된 zip파일의 압축을 해제하세요

    또는, `https://www.kaggle.com/datasets/saeedehkamjoo/standard-test-images`에서 데이터셋을 다운로드 한 후, dataset폴더에 넣을 수 있습니다.

### 실행 방법

1. **python 스크립트 실행**
    ```bash
    uv run python 스크립트-경로
    ```

2. **main.py 실행**
    main.py는 필수 argument로 configuration 경로를 받습니다. 다음과 같이 사용할 수 있습니다.
    ```bash
    uv run python main.py -c config-경로
    ```
    config 경로는 hydra의 최상위 config경로를 입력되면 되며, 기본적으로 `configs.config.yaml`에 있습니다.

    main.py는 선택 인자로 실행의 이름을 받습니다. 다음과 같이 사용하면, run directory의 뒤쪽에 이름이 붙습니다.
    ```bash
    uv run python main.py -c config-경로 -rn 실행-이름
    ```

3. **3D mesh INR 학습 실행 (`main_3d.py`)**
    별도 `pytorch3d` 설치 없이 실행할 수 있습니다.
    ```bash
    uv run python main_3d.py -c configs/config_3d.yaml -rn mesh-exp
    ```
    OBJ 경로는 `configs/learn_3d/learn_mesh.yaml`의 `mesh_path`로 설정하거나 오버라이드할 수 있습니다:
    ```bash
    uv run python main_3d.py -c configs/config_3d.yaml learn_3d.mesh_path=dataset/meshes/my_mesh.obj
    ```

### 설정(Configuration)
* 메인 설정 파일은 `configs/config.yaml`에 위치해 있습니다. 실험에 사용할 파라미터를 설정하려면 이 파일을 수정하세요.
* 하위 설정 파일은 `configs/`폴더의 하위 폴더에 저장되어 있습니다.
### 새로운 모델 또는 학습 로직 구현하기
* 새로운 모델이나 학습 로직을 구현하려면 `src/models/` 디렉토리에 새로운 클래스를 생성하세요. 일관성을 유지하기 위해 기존 구조를 따르면 좋습니다.
* 새로운 모델이나 학습 로직을 실행하려면, `main.py`를 수정하는 대신 `configs/config.yaml`또는 그 하위 폴더의 내용을 추가하거나 수정하세요.
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
