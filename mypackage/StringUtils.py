import math

def format_significant(value):
    if value == 0:
        return "0.0"
    else:
        # 有効桁数を求めるために対数を取る
        digits = -int(math.floor(math.log10(abs(value))))
        # 最小有効桁数を 1 とする
        digits = max(1, digits)
        # f-string を用いて動的にフォーマットを作成
        formatted = f"{value:.{digits}f}"
        return formatted