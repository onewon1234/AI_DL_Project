#!/bin/bash

echo "🎵 감정 분석 기반 노래 추천 시스템 설치 스크립트"
echo "=" * 50

# Python 버전 확인
python_version=$(python3 --version 2>/dev/null || echo "Python3 not found")
echo "Python 버전: $python_version"

# pip 확인
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip이 설치되어 있지 않습니다. pip을 먼저 설치해 주세요."
    exit 1
fi

# pip 명령 결정
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

echo "사용할 pip 명령: $PIP_CMD"

# requirements.txt가 있는지 확인
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt 파일을 찾을 수 없습니다."
    exit 1
fi

echo ""
echo "📦 패키지 설치 시작..."

# 기본 패키지 업그레이드
echo "1. pip 업그레이드..."
$PIP_CMD install --upgrade pip

# requirements.txt의 패키지들 설치
echo "2. 필수 패키지 설치..."
$PIP_CMD install -r requirements.txt

# 설치 확인
echo ""
echo "✅ 설치 완료! 패키지 버전 확인:"
echo "------------------------"

# 주요 패키지 버전 확인
python3 -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except ImportError:
    print('PyTorch: ❌ 설치되지 않음')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: ❌ 설치되지 않음')

try:
    import sentence_transformers
    print(f'Sentence-Transformers: {sentence_transformers.__version__}')
except ImportError:
    print('Sentence-Transformers: ❌ 설치되지 않음')

try:
    import faiss
    print(f'FAISS: {faiss.__version__}')
except ImportError:
    print('FAISS: ❌ 설치되지 않음')

try:
    import pandas as pd
    print(f'Pandas: {pd.__version__}')
except ImportError:
    print('Pandas: ❌ 설치되지 않음')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError:
    print('NumPy: ❌ 설치되지 않음')

try:
    import sklearn
    print(f'Scikit-learn: {sklearn.__version__}')
except ImportError:
    print('Scikit-learn: ❌ 설치되지 않음')
"

echo ""
echo "🎯 설치가 완료되었습니다!"
echo ""
echo "사용법:"
echo "  기본 실행: python3 main.py --num_samples 100 --num_epochs 3"
echo "  데모 실행: python3 demo.py --test_mode"
echo "  도움말:   python3 main.py --help"
echo ""
echo "자세한 사용법은 README.md 파일을 참고하세요."