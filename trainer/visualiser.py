# import os.path as osp
#
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
#
# class TrainVisualiser:
#     def __init__(self, logdir):
#         self.logdir = logdir
#
#     def __call__(self):
#         # plot training loss
#         self._performance_logs(osp.join(self.logdir, 'performance_log.csv'))
#
#         for phase in ['val', 'test']:
#             self._confusion_matrix(phase, osp.join(self.logdir, '{}-outputs.csv'.format(phase)))
#             # self._domain_confusion()
#
#     def _performance_logs(self, csv_path):
#         df = pd.read_csv(csv_path)
#         # Train loss
#         df.plot(y='train_loss', use_index=True)
#         plt.grid()
#         plt.title('Training Loss Performance Log')
#         plt.xlabel('Epoch')
#         plt.ylabel('Cross Entropy Loss')
#         plt.savefig(osp.join(self.logdir, 'train-loss'))
#
#         # Accuracy Plots
#         df.plot(y=[i for i in df.columns if 'acc' in i], use_index=True)
#         plt.grid()
#         plt.title('Accuracy Performance Log')
#         plt.xlabel('Epoch')
#         plt.ylabel('Classification Accuracy')
#         self.plot_max_vals(df)
#         plt.legend(loc='best')
#         plt.savefig(osp.join(self.logdir, 'acc-plots'))
#
#     def plot_max_vals(self, df):
#         for phase in ['val', 'test']:
#             idx = df[phase + '_acc'].argmax()
#             plt.axvline(x=idx, label=None, c='k', linestyle='--')
#             plt.scatter(idx, df['test_acc'][idx],
#                         label='{}max_testacc={:.3f}'.format(phase, df['test_acc'][idx]),
#                         marker='x')
#
#     def _confusion_matrix(self, phase, csv_path):
#         # Confusion matrix
#         df = pd.read_csv(csv_path)
#         conf_matrix = confusion_matrix(df['labels'], df['cls_pred'])
#         plt.figure()
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         sns.heatmap(conf_matrix, annot=True)
#         plt.savefig(osp.join(self.logdir, '{}-confusion-matrix'.format(phase)))
#
#         # Class histogram
#         plt.figure()
#         plt.grid()
#         sns.countplot(x=df['labels'])
#         plt.xlabel('Class Label')
#         plt.ylabel('Count')
#         plt.savefig(osp.join(self.logdir, '{}-class-histogram'.format(phase)))
#
#     def _domain_confusion(self):
#         raise NotImplementedError
