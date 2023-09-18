import os
import clip
import yaml
import torch
import pathlib
import argparse
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from awareness import awareness
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def start_train(args):

	ENCODER_PATH = args.encoder_path
	WINDOW_SIZE = args.window_size
	NUM_CLASSES = args.num_classes
	BATCH_SIZE = args.batch_size
	EPOCHS = args.epochs
	IMG_SIZE = args.img_size
	DYNAMIC_RAY = args.dynamic_ray
	MODEL_NAME = args.model_name
	DATASET_NAME = args.dataset_name

	encoder_model = torch.load(ENCODER_PATH)
	encoder_model.eval().to(device)

	feature_extractor = create_feature_extractor(encoder_model, return_nodes=['encoder'])

	awareness_model = awareness.Awareness(learnable=True, dynamic_ray=True)
	awareness_model.to(device)

	print("Encoder model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in encoder_model.parameters()]):,}")
	print("Awareness model parameters:", f"{int(np.sum([int(np.prod(p.shape)) for p in awareness_model.parameters()])):,}")

	preprocess = transforms.Compose(
		[transforms.Resize((IMG_SIZE,IMG_SIZE)),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)

	cifar10_train = CIFAR10(os.path.expanduser("~/.cache"), train=True, transform=preprocess, download=True)
	cifar10_test = CIFAR10(os.path.expanduser("~/.cache"), train=False, transform=preprocess, download=True)

	test_loader = DataLoader(
		cifar10_test,
		batch_size=int(BATCH_SIZE),
		shuffle=True
	)

	save_path = f'./checkpoints/{MODEL_NAME}_{DATASET_NAME}'.lower()

	config = {
		'model_name': MODEL_NAME,
		'encoder_path': ENCODER_PATH,
		'dataset': DATASET_NAME,
		'batch_size': BATCH_SIZE,
		'epochs': EPOCHS,
		'img_size': IMG_SIZE,
		'num_classes': NUM_CLASSES
	}

	results_data = {
		'epoch': [], 
		'ref_instances': [],
		'train_acc': [], 
		'test_acc': []
	}

	res_df = pd.DataFrame(results_data)

	if(not os.path.exists(f'{save_path}/')):
		os.makedirs(f'{save_path}')

	with open(f'{save_path}/config.yaml', 'w') as yaml_file:
		yaml.dump(config, yaml_file, default_flow_style=False)

	best_test_acc = 0.0
		
	for epoch in range(EPOCHS):
		
		train_loader = DataLoader(
			cifar10_train,
			batch_size=BATCH_SIZE,
			shuffle=True
		)
		
		awareness_model.__init__(learnable=True, dynamic_ray=True)
		
		encoder_model.eval()
		awareness_model.eval()
		
		train_loaders = [train_loader]
		
		with torch.no_grad(): 
		
			for train_loader in train_loaders:
				for i, (images, labels) in enumerate(train_loader):
					
					train_correct_preds_batches = []
					test_correct_preds_batches = []

					train_count = 0
					test_count = 0

					if torch.cuda.is_available():
						images = Variable(images.cuda())
						labels = Variable(labels.cuda())

					#features = encoder_model(images).float()
					features = torch.mean(feature_extractor(images)['encoder'].float(), 1)
					
					preds = awareness_model(torch.unsqueeze(features,1), set_labels=labels, update_ref_insts=True)

					train_correct_preds_batch = np.sum(preds.cpu().numpy() == labels.cpu().numpy())
					train_correct_preds_batches.append(train_correct_preds_batch)
					train_count = train_count+len(images)

					references = awareness_model.awareness.ref_insts
					references_labels = awareness_model.awareness.ref_insts_labels

					n_ref_insts = len(references)

					train_acc = round(np.sum(train_correct_preds_batches)/train_count, 4)

					print(f'Train ({n_ref_insts} refs) --> {train_acc}')
			
			print('####')
			for i, (images, labels) in enumerate(test_loader):

				if torch.cuda.is_available():
					images = Variable(images.cuda())
					labels = Variable(labels.cuda())

				#features = encoder_model(images).float()
				features = torch.mean(feature_extractor(images)['encoder'].float(), 1)
				
				preds = awareness_model(torch.unsqueeze(features,1))

				test_correct_preds_batch = np.sum(preds.cpu().numpy() == labels.cpu().numpy())
				test_correct_preds_batches.append(test_correct_preds_batch)
				test_count = test_count+len(images)

				test_acc = round(np.sum(test_correct_preds_batches)/test_count, 4)

				print(f'Test ({n_ref_insts} refs) --> {test_acc}')

			print(f'Epoch {epoch+1}, Reference instances (N): {n_ref_insts}, Train accuracy: {train_acc}, Test accuracy: {test_acc}')

			results_data = {
				'epoch': epoch+1, 
				'ref_instances': n_ref_insts, 
				'train_acc': train_acc, 
				'test_acc': test_acc
			}

			if(not os.path.exists(f'{save_path}/weights')):
				os.makedirs(f'{save_path}/weights')
		
			res_df.loc[len(res_df)] = results_data
			res_df.to_csv(f'{save_path}/results.csv', index=False)
		
			torch.save(awareness_model, f'{save_path}/weights/last.pt')
		
			if(test_acc > best_test_acc):
				torch.save(awareness_model, f'{save_path}/weights/best.pt')
				best_test_acc = test_acc
		
				print(f'Saved checkpoint related to better accuracy score: {best_test_acc}')


	pass

def main(args):
	start_train(args)
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Your Script Description')
	parser.add_argument('--encoder_path', type=str, default='./checkpoints/vit-b-16_cifar-10/experiments/exp1_tl0.0082_ta0.9526/weights/best.pt', help='Path to the encoder model checkpoint')
	parser.add_argument('--window_size', type=int, default=1, help='Window size')
	parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--img_size', type=int, default=224, help='Image size')
	parser.add_argument('--dynamic_ray', type=bool, default=True, help='Use dynamic ray')
	parser.add_argument('--model_name', type=str, default='Awareness+ViT-B-16', help='Model name')
	parser.add_argument('--dataset_name', type=str, default='CIFAR-10', help='Dataset name')

	args = parser.parse_args()
	main(args)