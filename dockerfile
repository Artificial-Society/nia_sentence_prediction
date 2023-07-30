# 기본 이미지로 PyTorch가 설치된 Python 이미지를 사용
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지를 설치
RUN pip install torch transformers scikit-learn tensorboard ml-things pandas

# 현재 디렉토리의 모든 파일을 /app에 복사
COPY . /app

# 기본 명령 실행
CMD ["python", "inference.py"]

