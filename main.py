from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict() # leave dict empty; no data augmentation used for training; 
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True) 
model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1) 
saveResult("data/membrane/test",results)
