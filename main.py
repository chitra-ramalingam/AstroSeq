from src.Classifiers.CnnModel import CnnModel

def main():

    
    cnnModel = CnnModel()
    #this creates the .keras model file
    #cnnModel.runAstro1DCNN()
    # this one caches the star segments in lccache and then does star-based prediction
    # the star scores are saved to starwise_score_1dcnn.csv
    # the better the scrore implies the higher the chance of an exoplanet transit
  
    # cnnModel.runStarbased1DCNN()
   # cnnModel.runStarVecEmbeddings()
    cnnModel.runTestOnStarVecEmbeddings()
    

if __name__ == "__main__":
    main()