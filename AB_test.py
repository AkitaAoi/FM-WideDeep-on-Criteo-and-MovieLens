import json
from scipy import stats

import config
from FM import cross_validate_fm
from WideDeep_on_MovIeLens import cross_validate_widedeep

def main():
    print("开始 FM 5折交叉验证...")
    fm_aucs = cross_validate_fm(file_path = config.MOVIELENS_DIR, n_splits=5)
    print(f"FM AUC: {fm_aucs}")

    print("\n开始 WideDeep 5折交叉验证...")
    wd_aucs = cross_validate_widedeep(file_path=config.MOVIELENS_DIR, n_splits=5)
    print(f"WideDeep AUC: {wd_aucs}")

    # 保存结果到 JSON
    results = {
        'fm': fm_aucs.tolist() if hasattr(fm_aucs, 'tolist') else fm_aucs,
        'widedeep': wd_aucs.tolist() if hasattr(wd_aucs, 'tolist') else wd_aucs
    }
    with open('auc_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n结果已保存到 auc_results.json")

    # 可选：直接进行 t 检验
    t_stat, p_value = stats.ttest_rel(wd_aucs, fm_aucs)
    print(f"\n配对 t 检验: t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print("结论: WideDeep 显著优于 FM (p < 0.05)")
    else:
        print("结论: 未检测到显著差异")

if __name__ == '__main__':
    main()