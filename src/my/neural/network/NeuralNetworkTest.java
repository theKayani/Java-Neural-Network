package my.neural.network;

import java.util.Arrays;

import my.neural.network.NeuralNetwork.ActivationFunction;

public class NeuralNetworkTest
{
	public static void main(String[] args)
	{
		NeuralNetwork brain = new NeuralNetwork(2, 2, 1);
		brain.setLearningRate(0.05);
		brain.setActivationFunction(new ActivationFunction(Mat.TANH, Mat.TANH_DERIVATIVE));
		
		int[][] xorInputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		for(int i = 0; i < xorInputs.length * 2000; i++)
		{
			int[] arr = xorInputs[i % 4];
			double[] inputs = { arr[0], arr[1] };
			double[] outputs = { arr[0] ^ arr[1] };
			brain.train(inputs, outputs);
		}
		
		for(int i = 0; i < xorInputs.length; i++)
		{
			double[] inputs = { xorInputs[i][0], xorInputs[i][1] };
			double[] outputs = brain.process(inputs);
			System.out.println("Test: " + Arrays.toString(inputs) + " -> " + Arrays.toString(outputs));
		}
	}
}
