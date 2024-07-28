from Core.data import Data
from Core.model import Model

data = Data()
data.read('G:/LSTM_personal/MYDATA/Darjeeling_daily_10y.csv')
data.check_null()
data.clean_dataset()
data.plot_train_points(col = 'NORMAL (mm)')
data.plot_train_points(col = 'ACTUAL (mm)')
data.print_head()
data.print_desc()

trainer = Model(data.df)
trainer.prep_data(feature_col = 'NORMAL (mm)', step = 10)
trainer.plot_train_test(Tp = 2894)
trainer.call_for_conv_mat(step = 10)
trainer.call_for_build()
trainer.model_training()
trainer.model_loss_curve()
trainer.model_saw_plot()
trainer.model_predicted_plot()
trainer.groundTruth_prediction_plot()
trainer.model_evaluation()