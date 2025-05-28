# Deep Learning Study

딥러닝 학습을 위한 프로젝트입니다.

## 프로젝트 구조

```
.
├── src/                    # 소스 코드
│   ├── data/              # 데이터 관련 코드
│   ├── models/            # 모델 정의
│   └── utils/             # 유틸리티 함수
├── notebooks/             # Jupyter 노트북
├── data/                  # 데이터셋
└── requirements.txt       # 프로젝트 의존성
```

## 설치 방법

1. Python 3.8 이상이 설치되어 있어야 합니다.
2. 가상환경 생성 및 활성화:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. Jupyter Notebook 실행:
   ```bash
   jupyter notebook
   ```
2. `notebooks` 디렉토리에서 학습을 시작할 수 있습니다. 