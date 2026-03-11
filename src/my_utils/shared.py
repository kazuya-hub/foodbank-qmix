"""
実験コード群の異なるファイル間で変数を共有するための、単なる名前空間

使い方:
    import my_utils.shared as shared
    
    # 書き込み
    shared.log_q_values = False
    shared.run_count = 0
    
    # 読み込み
    if shared.log_q_values:
        ...
"""
