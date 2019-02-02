using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    class NeuralNetwork
    {
        /// <summary>
        /// Array of all layer sizes for this Neural Network
        /// </summary>
        int[] layerSizes;

        /// <summary>
        /// Array of all layers (excluding the input layer) for this Neural Network
        /// </summary>
        Layer[] layers;

        /// <summary>
        /// Learning rate of all nn instances
        /// </summary>
        public static double LearningRate = .0020;
        
        /// <summary>
        /// Initialization method for the Neural Network
        /// </summary>
        /// <param name="_layerSizes">Array of all layer sizes for this Neural Network</param>
        public NeuralNetwork(int[] _layerSizes)
        {
            // set the layer sizes 
            layerSizes = _layerSizes;

            // initiate the layers based on layer count and sizes
            layers = new Layer[layerSizes.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                if (i == layers.Length - 1)
                {
                    // create the output layer given it's input and output size
                    layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], Layer.ActivationFunction.Softmax);
                }

                // create a hidden layer given it's input and output size
                layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], Layer.ActivationFunction.ReLU);
            }
        }

        /// <summary>
        /// Compute all layers of the nn
        /// </summary>
        /// <param name="inputs">Array of inputs that the nn should use to ompute layers</param>
        /// <returns>Returns the output of the network</returns>
        public double[] ComputeNN(double[] inputs)
        {
            // compute the first layer using inputs
            layers[0].FeedForward(inputs);

            // compute the rest of the layers using previous layers in series
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward(layers[i - 1].outputs);
            }

            // return the output of the last layer (output layer)
            return layers[layers.Length - 1].outputs;
        }

        /// <summary>
        /// Adjust weights and biases based on target input
        /// </summary>
        /// <param name="target"></param>
        public void Learn(double[] target)
        {
            // compute back propagation on all layers
            for (int i = layers.Length - 1; i >= 0; i--) 
            {
                if (i == layers.Length - 1) 
                {
                    // compute back propagation of output layer
                    layers[i].BackPropOutput(target);
                }
                else
                {
                    // compute back propagation of hidfden layer
                    layers[i].BackPropHidden(layers[i + 1].error, layers[i + 1].weights);
                }
            }

            // update weights of all layers
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights();
            }
        }

        /// <summary>
        /// Gets the combined error (loss) of the output layer
        /// </summary>
        /// <returns></returns>
        public double GetLoss()
        {
            return layers[layers.Length - 1].error.Sum();
        }
    }
}
