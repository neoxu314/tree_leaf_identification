import matplotlib.pyplot as plt
import train_cnn as tc
import numpy as np

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

X_train, X_test, y_train, y_test = tc.get_dataset(load=False)
model, opt = tc.create_model_two_conv(X_train.shape[1:])

print(X_train.shape[1:])

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model, history = tc.training(model, X_train, X_test, y_train, y_test, data_augmentation=False)

# visualisation for model
plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# visualisation for loss and accuracy
plt.plot()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save accuracy data in text
accy = history.history['acc']
np_accy = np.array(accy)
np.savetxt('./save_data/save.txt', np_accy)
