import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Configurações de estilo avançadas
plt.style.use('seaborn-v0_8-talk')  # Estilo mais moderno e adequado para apresentações
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.labelpad'] = 12

# Paleta de cores profissional
cores = ['#2E86AB', '#F18F01']  # Azul científico e laranja de destaque
cores_texto = ['#333333', '#333333']

# Dados
groups = ["1-20", "20-40", "40-60", "60-80", "80-100"]
puzzlecam_iou = [37.82, 59.79, 70.78, 81.23, 94.7]
puzzlecam_loss_iou = [34.8, 60.25, 72.08, 82.32, 94.71]

# Configuração da figurat
fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
fig.patch.set_facecolor('#F5F5F5')  # Fundo suave
ax.set_facecolor('#FFFFFF')  # Fundo branco para o gráfico

# Posicionamento das barras
x = np.arange(len(groups))
width = 0.35  # Barras mais largas para melhor visualização

# Plot das barras com efeito de sombra
bars1 = ax.bar(x - width/2, puzzlecam_iou, width, 
               label='PuzzleCAM (Baseline)', 
               color=cores[0], edgecolor='white', linewidth=1.5,
               alpha=0.95, zorder=3)

bars2 = ax.bar(x + width/2, puzzlecam_loss_iou, width, 
               label = r'PuzzleCAM + $\mathcal{L}_{icu}$ ($\sigma=2$)', 
               color=cores[1], edgecolor='white', linewidth=1.5,
               alpha=0.95, zorder=3)

# Adicionando linhas de conexão para melhor comparação
for i in range(len(x)):
    ax.plot([x[i]-width/2, x[i]+width/2], 
             [puzzlecam_iou[i], puzzlecam_loss_iou[i]], 
             color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

# Personalização dos eixos
ax.set_ylabel('IoU Score (%)', fontsize=13, fontweight='bold', color=cores_texto[0])
ax.set_xlabel('Percentage of Burned Area in Image', fontsize=13, fontweight='bold', color=cores_texto[0])
ax.set_title('Comparative Performance on "Burned" Class Segmentation\nby Burned Area Percentage Range', 
             fontsize=15, fontweight='bold', pad=25, color=cores_texto[0])

# Configuração dos ticks
ax.set_xticks(x)
ax.set_xticklabels([f"{group}%" for group in groups], fontsize=12)
ax.set_ylim(0, 105)  # Espaço para os rótulos
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='both', which='major', labelsize=11)

# Grade estilizada
ax.grid(axis='y', linestyle=':', alpha=0.6, color='gray', zorder=0)

# Adicionando valores nas barras com formatação melhorada
def autolabel(bars, color):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    color=color)

autolabel(bars1, cores[0])
autolabel(bars2, cores[1])

# Legenda estilizada
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=2, fontsize=12, framealpha=1, 
                   facecolor='white', edgecolor='#DDDDDD')
legend.get_frame().set_linewidth(1.2)

# Adicionando nota explicativa
# plt.figtext(0.5, 0.001, 
#             'Evaluation on Fire Dataset (10,000 samples) | Higher IoU indicates better segmentation accuracy',
#             ha='center', fontsize=10, style='italic', color='#555555')

# Ajustes finais
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Espaço para a legenda

# Salvar em alta qualidade
plt.savefig('graphic.png', bbox_inches='tight', dpi=300)
print("Saved graphic.png")