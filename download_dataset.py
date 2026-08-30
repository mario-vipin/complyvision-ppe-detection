
from roboflow import Roboflow
rf = Roboflow(api_key="pGLGEngymdrXhVzw9Dub")
project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
version = project.version(30)
dataset = version.download("yolov8")
                