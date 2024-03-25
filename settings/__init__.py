from .Options import Options
# from .MainOptions import TrainOptions, TestOptions









def Option(mode):
   if mode == "train":
      return Options().get_options()
