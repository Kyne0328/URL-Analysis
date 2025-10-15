import matplotlib.pyplot as plt
import numpy as np

# Test the color formats
fig, ax = plt.subplots()
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#0f0f0f')

# Test the fixed colors
ax.grid(True, alpha=0.3, color=(1.0, 1.0, 1.0, 0.3))
ax.text(0.5, 0.5, 'Test', color='white',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor=(0.4, 0.494, 0.918, 0.2),
                  edgecolor=(0.4, 0.494, 0.918, 0.4),
                  alpha=0.8))

plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print('Color formats work correctly!')
