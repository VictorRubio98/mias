import matplotlib.pyplot as plt
import pandas as pd

pg = pd.read_csv('code/pg.txt')
pg = pg.iloc[0:3]
pg = pg[pg.columns[0:6]]
plot_pg = pd.DataFrame([], columns=['mean','min','max'])
plot_pg['mean'] = pg.T.mean()
plot_pg['min'] = abs(pg.T.min()-pg.T.mean())
plot_pg['max'] = abs(pg.T.max()-pg.T.mean())
assym = [plot_pg['min'].values, plot_pg['max'].values]
plt.errorbar(plot_pg.index, plot_pg['mean'].values, yerr=assym, fmt='b*', elinewidth=2)
plt.plot(plot_pg.index, plot_pg['mean'].values, label='Media de la ganancia de la privacidad')
plt.plot(plot_pg.index, [0.63, 0.63, 0.63], '--', c='r', label='Ventaja sin privacidad')
ax = plt.gca()
ax.legend()
plt.ylim(0.5,1)
plt.title('Ganancia de privacidad')
plt.xlabel('Épsilon')
for a, p in zip(plot_pg.index, plot_pg.values):
    ax.annotate(f'{p[0]:.2f} - {p[1]:.2f} + {p[2]:.2f}', (a, p[0]), textcoords='offset points',
                xytext=(0, 3), ha='center', va='bottom', fontsize='x-small')

plt.savefig('./code/plot_fig.png')