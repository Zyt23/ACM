import pandas as pd

ac_320_new = [ 
        "B-301A", "B-301G", "B-301H", "B-301J", "B-301K", "B-301L", "B-301W", "B-301Y",
        "B-302J", "B-302K", "B-302L", "B-305C", "B-305D", "B-305E", "B-305M", "B-305N",
        "B-305P", "B-305Q", "B-306G", "B-306M", "B-307R", "B-307S", "B-309K", "B-309L",
        "B-309M", "B-309X", "B-309Y", "B-30A8", "B-30AA", "B-30AJ", "B-30AK", "B-30CC",
        "B-30CD", "B-30DG", "B-30EZ", "B-30F8", "B-30FX", "B-322F", "B-322V", "B-323E",
        "B-325D", "B-32ET", "B-32EU", "B-32EV", "B-32FU", "B-32FV", "B-32FW", "B-32FX",
        "B-32FY", "B-32GC", "B-32GD", "B-32GG", "B-32H3", "B-32HQ", "B-32J6", "B-32JL",
        "B-32L2", "B-32LL", "B-32LN", "B-32LP", "B-32LR", "B-32M8", "B-32ML", "B-32MQ"
    ]
if __name__ == "__main__":
    fault_df = pd.read_csv("320_ACM_faults.csv")
    fault_df = fault_df.dropna(subset=["机号"])
    fault_tails = set(fault_df["机号"].unique())
    ac_set = set(ac_320_new)

    inter = sorted(fault_tails & ac_set)
    only_in_fault = sorted(fault_tails - ac_set)
    only_in_ac = sorted(ac_set - fault_tails)

    print("=== 既在 ac_320_new 又在 故障表里的机号（真正会被当成故障机来处理） ===")
    print(inter)

    print("\n=== 只在故障表里，不在 ac_320_new 里的机号（你这批数据里用不到） ===")
    print(only_in_fault)

    print("\n=== 只在 ac_320_new 里，从未在故障表出现过的机号（永远当正常机） ===")
    print(only_in_ac)