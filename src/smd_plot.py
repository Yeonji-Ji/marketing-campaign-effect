from typing import Optional
import matplotlib.pyplot as plt

def smd_plot(results,              
             title: str = 'Covariate Balance: Before & After PSM',
             save_file: Optional[str] = None,
             show_plot: bool = True):
    
    smd_before = results['smd_before_detail']
    smd_after = results['smd_after_detail']
    covariates = list(smd_before.keys())
    y_pos = np.arange(len(covariates))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(smd_before.values(), y_pos, color='red', label=f"Before Matching (mean={results['smd_before_mean']:.3f})", s=80)
    ax.scatter(smd_after.values(), y_pos, color='blue', label=f"After Matching (mean={results['smd_after_mean']:.3f})", s=80)
    
    for i in range(len(covariates)):
        ax.plot([list(smd_before.values())[i], list(smd_after.values())[i]], [y_pos[i], y_pos[i]], color='grey', linestyle='--')
    
    ax.axvline(x=0.1, color='black', linestyle=':', linewidth=1)
    ax.text(0.04, -0.8, '   Balance\nThreshold (0.1)', color='black', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.invert_yaxis()
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance: Before & After PSM', fontsize=16)
    ax.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_file:
        plt.savefig(FIG_PATH + '/'+ save_file, dpi=200, bbox_inches='tight')
        print(f"Plot saved as {save_file}")

    if show_plot:
        plt.show()