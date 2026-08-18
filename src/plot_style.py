"""matplotlib에서 한글이 네모(□)로 깨지지 않게 폰트를 설정하는 공통 모듈.

matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어서, 차트 제목/축
라벨/범례에 한글을 쓰면 깨진다. 저장소에 폰트 파일(assets/fonts/)을
함께 포함해서, 시스템에 한글 폰트가 없는 환경(다른 사람의 컴퓨터, CI 등)
에서 재현해도 항상 동일하게 렌더링되도록 한다.

사용법: 각 시각화 스크립트에서
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
세 줄 대신 아래 한 줄만 쓰면 된다.
    from plot_style import plt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NanumGothic.ttf"
if _FONT_PATH.exists():
    fm.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스(-) 기호가 깨지는 것 방지

__all__ = ["plt"]
