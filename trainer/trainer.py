import os

import torch
import numpy as np
import pandas as pd


class Trainer:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = None
        self.train_loss = []
        self.train_acc = []
        self.valtest_acc = {'test': [], 'val': []}

    def train(self, model, criterion, optimizer, lr_scheduler, train_loader, test_loaders, epochs=30):
        for ep in range(epochs):
            # train epoch
            model, epoch_loss, epoch_acc = self._train_epoch(model, train_loader, criterion, optimizer)

            # validate/test model
            self.test(model, test_loaders)

            # write to tensorboard
            self.write_tensorboard()

            # update performance history
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)

            # Step scheudler
            if lr_scheduler is not None:
                lr_scheduler.step()

        self.save_logs()

    @staticmethod
    def _train_epoch(model, train_loader, criterion, optimizer):
        model.train()
        epoch_loss = 0.
        epoch_acc = 0.
        if torch.cuda.is_available():
            model = model.cuda()

        for image_paths, images, labels in train_loader:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            class_logits = model(images)
            loss = criterion(class_logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, cls_preds = class_logits.max(dim=1)
            epoch_acc += torch.sum((cls_preds == labels.data).float()).item()

        # Average over training dataset for the epoch
        epoch_acc /= len(train_loader.dataset)
        epoch_loss /= len(train_loader.dataset)
        return model, epoch_loss, epoch_acc

    def test(self, model, test_loaders):
        model.eval()
        with torch.no_grad():
            for phase, loader in test_loaders.items():
                total = len(loader.dataset)
                class_correct, all_preds = self._test_epoch(model, loader)
                class_acc = float(class_correct) / total
                self.valtest_acc[phase].append(class_acc)
                if class_acc >= max(self.valtest_acc[phase]):
                    self.store_outputs(all_preds, phase)

    @staticmethod
    def _test_epoch(model, loader):
        class_correct = 0.
        all_preds = {'image_paths': [],
                     'labels': [],
                     'cls_pred': []}
        for image_paths, data, labels in loader:
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            class_logit = model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum((cls_pred == labels.data).float()).item()

            # attach all preds
            all_preds['image_paths'].extend(list(image_paths))
            all_preds['labels'].extend(labels.tolist())
            all_preds['cls_pred'].extend(cls_pred.tolist())
        return class_correct, all_preds

    def store_outputs(self, all_preds, phase):
        df = pd.DataFrame(all_preds)
        save_path = os.path.join(self.logdir, 'outputs', '{}-outputs.csv'.format(phase))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    def write_tensorboard(self):
        # Tensorboard reports aren't going atm because the GPU port clusters have changed and it no longer works
        pass

    def save_logs(self):
        df = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_acc': self.valtest_acc['val'],
            'test_acc': self.valtest_acc['test']
              }

        df = pd.DataFrame(df)
        save_path = os.path.join(self.logdir, 'outputs', 'performance_log.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    def print_report(self, rep):
        best_idx = np.array(self.valtest_acc['val']).argmax()
        best_cv = self.valtest_acc['test'][best_idx]
        best_test = np.max(self.valtest_acc['test'])
        print('\tRep-{}:\tBest CV Accuracy @ Epoch-{}={:.3f} '
              '\tvs.\tBest overall Test accuracy={:.3}'.format(rep, best_idx, best_cv, best_test))
