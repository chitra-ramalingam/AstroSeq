from src.Classifiers.K2.Pipeline.K2CampaignRunner import K2CampaignRunner
from src.Classifiers.CnnModel import CnnModel
from src.Classifiers.LargeWindow.LargeWindow_Processor import LargeWindowCnnModel
def main():
    K2CampaignRunner().run()


def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)

if __name__ == "__main__":
    main()