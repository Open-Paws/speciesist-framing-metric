import evaluate
from evaluate.utils import launch_gradio_widget
module = evaluate.load("speciesist_framing")
launch_gradio_widget(module)
