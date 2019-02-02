using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    class Layer
    {
        int inputCount;
        int outputCount;

        public double[] inputs;
        public double[] outputs;

        public double[,] weights;
        public double[] biases;

        public double[] gamma;
        public double[] error;

        public double[,] weightsDelta;
        public double[] biasesDelta;

        public double[,] previousWeightsDelta;

        double learningRate;
        double momentum;

        public enum ActivationFunction { ReLU, Softmax };
        delegate double[] Activation(double[] input);
        Activation activate;
        Activation derivActivate;

        Random random = new Random();

        public Layer(int _inputCount, int _outputCount, ActivationFunction activationFunction)
        {
            inputCount = _inputCount;
            outputCount = _outputCount;

            inputs = new double[inputCount];
            outputs = new double[outputCount];

            weights = new double[outputCount, inputCount];
            biases = new double[outputCount];

            error = new double[outputCount];
            gamma = new double[outputCount];

            weightsDelta = new double[outputCount, inputCount];
            biasesDelta = new double[outputCount];

            learningRate = NeuralNetwork.LearningRate;

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

            InitializeWeights();
        }

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

        void InitializeBiases()
        {
            for (int i = 0; i < outputCount; i++)
            {
                biases[i] = 1;
            }
        }

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

        public void BackPropOutput(double[] target)
        {
            for (int i = 0; i < outputCount; i++)
            {
                error[i] = outputs[i] - target[i];
            }

            double[] deriv = derivActivate(outputs);
            for (int i = 0; i < outputCount; i++)
            {
                gamma[i] = error[i] * deriv[i];
            }

            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }
        }

        public void BackPropHidden(double[] gammaForward, double[,] weightsForward)
        {
            double[] deriv = derivActivate(outputs);
            for (int i = 0; i < outputCount; i++)
            {
                gamma[i] = 0;
                for (int j = 0; j < gammaForward.Length; j++)
                {
                    gamma[i] += gammaForward[j] * weightsForward[j, i];
                }

                gamma[i] *= deriv[i];
            }

            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < outputCount; i++)
            {
                for (int j = 0; j < inputCount; j++)
                {
                    weights[i, j] -= weightsDelta[i, j] * learningRate;
                }
            }
        }

        double[] ReLU(double[] x)
        {
            double[] result = new double[x.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Math.Max(0, x[i]);
            }
            return result;
        }

        double[] DeriveReLU(double[] x)
        {
            double[] result = new double[x.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (x[i] <= 0) ? 0 : 1;
            }
            return result;
        }

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

        double[] DeriveSoftmax(double[] x)
        {
            return (double[])x.Select(i => (1 - i) * i);
        }
    }
}
