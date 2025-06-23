import pydot
from graphviz import Source

src = Source.from_file('input.dot')
src.render(
    filename='output_image',
    format='png',
    engine="dot",
    cleanup=True
# args=["-Gsize=6,6", "-Gdpi=1000"]
)

# # 读取 DOT 文件
# graphs = pydot.graph_from_dot_file("input.dot")
#
# graph = graphs[0]
#
# # 生成 PNG 图片
# graph.write_png('output.png')