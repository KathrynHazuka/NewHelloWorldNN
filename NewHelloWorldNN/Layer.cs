using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    class Layer
    {
        // input count of the layer
        int inputCount;
        // output count of the layer
        int outputCount;

        /// <summary>
        /// Array of all inputs
        /// </summary>
        public double[] inputs;

        /// <summary>
        /// Array of all outputs
        /// </summary>
        public double[] outputs;

        /// <summary>
        /// 2D array of all weights
        /// </summary>
        public double[,] weights;

        /// <summary>
        /// Array of all biases
        /// </summary>
        public double[] biases;

        /// <summary>
        /// Error for gradient descent calculations
        /// </summary>
        public double[] error;

        /// <summary>
        /// Derivitive of the Cost function (1/2)(predY - y)^2
        /// </summary>
        public double[] derivCost;

        // delta of weights per gradient descent calculations
        double[,] weightGradient;
        // delta of biases per gradient descent calculations
        double[] biasesDelta;

        // learning rate of the layer
        double learningRate;
        // momentum for gradient descent calculations
        double momentum;

        /// <summary>
        /// What Activation Function should this layer use?
        /// </summary>
        public enum ActivationFunction { ReLU, Softmax };

        // delagate for Activation Functions
        delegate double[] Activation(double[] input);

        // Activation Function as a variable
        Activation activate;
        // derivitive of Activation Function as a variable
        Activation derivActivate;

        // random for generation random numbers
        Random random = new Random();

        /// <summary>
        /// A self contained Layer of the neural network
        /// </summary>
        /// <param name="_inputCount">Input count for this layer</param>
        /// <param name="_outputCount">Output count for this layer</param>
        /// <param name="activationFunction">Activation function this layer should use</param>
        public Layer(int _inputCount, int _outputCount, ActivationFunction activationFunction)
        {
            // set input and output counts
            inputCount = _inputCount;
            outputCount = _outputCount;

            // initialize all arrays based on input and output count
            inputs = new double[inputCount];
            outputs = new double[outputCount];

            weights = new double[outputCount, inputCount];
            biases = new double[outputCount];

            derivCost = new double[outputCount];
            error = new double[outputCount];

            weightGradient = new double[outputCount, inputCount];
            biasesDelta = new double[outputCount];

            // set learning rate based on NeuralNetowrk class
            learningRate = NeuralNetwork.LearningRate;

            // assign activation and derivitive functions
            switch (activationFunction)
            {
                case ActivationFunction.ReLU:
                    activate = ReLU;
                    derivActivate = DeriveReLU;
                    break;
                case ActivationFunction.Softmax:
                    activate = Softmax;
                    derivActivate = DeriveSoftmax;
                    break;
                default:
                    activate = ReLU;
                    derivActivate = DeriveReLU;
                    break;
            }

            // initialize all weights and biases
            InitializeWeights();
            InitializeBiases();
        }

        // initialize all weights using He-at-al
        void  InitializeWeights()
        {
            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weights[i,j] = random.NextDouble() * Math.Sqrt(2d / inputCount);
                }
            }
        }

        // initialize all biases to 0
        void InitializeBiases()
        {
            for (int i = 0; i < outputCount; i++)
            {
                biases[i] = 0;
            }
        }

        /// <summary>
        /// Computes this layer based on input.
        /// </summary>
        /// <param name="_inputs">Input from last layer</param>
        /// <returns>Return output of this layer</returns>
        public double[] FeedForward(double[] _inputs)
        {
            inputs = _inputs;

            for (int i = 0; i < outputCount; i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < inputCount; j++)
                {
                    outputs[i] += inputs[j] * weights[i, j];
                }
                outputs[i] += biases[i];
            }
            outputs = activate(outputs);

            return outputs;
        }

        /// <summary>
        /// Computes gradient descent using back propagation for the output layer (last layer) using target values
        /// </summary>
        /// <param name="target">Desired target output</param>
        public void BackPropOutput(double[] target)
        {
            // compute derivitive of the cost function in respect to outputs
            for (int i = 0; i < outputCount; i++)
            {
                derivCost[i] = outputs[i] - target[i];
            }

            // compute derivitive of the activation function in respect to outputs
            double[] derivActive = derivActivate(outputs);

            // compute error of layer
            for (int i = 0; i < outputCount; i++)
            {
                error[i] = derivCost[i] * derivActive[i];
            }

            // compute cost derivitive in respect to weights
            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weightGradient[i, j] = error[i] * inputs[j];
                }
            }
        }

        /// <summary>
        /// Computes gradient descent using back propagation for the hidden layers using forward layer information
        /// </summary>
        /// <param name="errorForward">Error of the forward later (i - 1)</param>
        /// <param name="weightsForward">weights of the forward layer (i - 1)</param>
        public void BackPropHidden(double[] errorForward, double[,] weightsForward)
        {
            // compute derivitive of the activation function in respect to outputs
            double[] derivActive = derivActivate(outputs);

            // compute error of this layer
            for (int i = 0; i < outputCount; i++)
            {
                error[i] = 0;
                for (int j = 0; j < errorForward.Length; j++)
                {
                    error[i] += errorForward[j] * weightsForward[j, i];
                }
                error[i] *= derivActive[i];
            }

            // compute cost derivitive in respect to weights
            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weightGradient[i, j] = error[i] * inputs[j];
                }
            }
        }

        /// <summary>
        /// Updated the weights based on weight gradient and learning rate
        /// </summary>
        public void UpdateWeights()
        {
            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weights[i, j] -= weightGradient[i, j] * learningRate;
                }
            }
        }

        // compute the ReLU Activation Function for each output
        double[] ReLU(double[] x)
        {
            double[] result = new double[x.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Math.Max(0, x[i]);
            }
            return result;
        }

        // compute the derivitive of ReLU Activation Function for each output
        double[] DeriveReLU(double[] x)
        {
            double[] result = new double[x.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (x[i] <= 0) ? 0 : 1;
            }
            return result;
        }

        // compute the Softmax Activation Function for each output
        double[] Softmax(double[] x)
        {
            double max = x[0];
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] > max) max = x[i];
            }

            double[] adjustedX = (double[])x.Select(i => Math.Exp(i - max));
            double scale = adjustedX.Sum();

            return (double[])adjustedX.Select(i => i / scale);
        }

        // compute the derivitive of Softmax Activation Function for each output
        double[] DeriveSoftmax(double[] x)
        {
            return (double[])x.Select(i => (1 - i) * i);
        }
    }
}
