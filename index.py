from roboflow import Roboflow
rf = Roboflow(api_key="4EDUd6Y8GFTj2h8n7Ayx")
project = rf.workspace("satvik-yadav-a0pyc").project("my-first-project-vpfeo-wgmqe")
version = project.version(2)
dataset = version.download("multiclass")
                