using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    class NewHelloWorldNN
    {
        static XOR[] xor;
        static Random random = new Random();

        static void Main(string[] args)
        {
            // create database of XOR inputs and outputs
            xor = new XOR[8];
            xor[0] = new XOR(new double[] { 0, 0, 0 }, new double[] { 0 });
            xor[1] = new XOR(new double[] { 1, 0, 0 }, new double[] { 1 });
            xor[2] = new XOR(new double[] { 0, 1, 0 }, new double[] { 1 });
            xor[3] = new XOR(new double[] { 0, 0, 1 }, new double[] { 1 });
            xor[4] = new XOR(new double[] { 1, 1, 0 }, new double[] { 0 });
            xor[5] = new XOR(new double[] { 1, 0, 1 }, new double[] { 0 });
            xor[6] = new XOR(new double[] { 0, 1, 1 }, new double[] { 0 });
            xor[7] = new XOR(new double[] { 1, 1, 1 }, new double[] { 0 });

            // create neural network based on layer sizes
            NeuralNetwork nn = new NeuralNetwork(new int[] { 3, 25, 25, 1 });

            Console.Write("---------------------------------------------------------------------\n");
            Console.Write("Training test to teach the Neural Network to evaluate XOR gate logic: \n");
            Console.Write("---------------------------------------------------------------------\n");

            // teach network based on random XOR operations
            for (int i = 0; i < 5000; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    int r = random.Next(0, 7);
                    nn.ComputeNN(xor[r].input);
                    nn.Learn(xor[r].target);
                }

                if (i % 200 == 0)
                {
                    // write loss after n itterations through the learning loop
                    double loss = nn.GetLoss();
                    Console.Write("Loss after " + (i * 20) + " samples " + loss.ToString("F8") + "\n");
                }
            }
            Console.Write("\n");
            Console.Write("--------------------------\n");
            Console.Write("Outputs using Testing Set:\n");
            Console.Write("--------------------------\n");

            // test network using random XOR inputs
            for (int i = 0; i < 20; i++)
            {
                int r = random.Next(0, 7);
                Console.Write(String.Format("Input {0} | Output {1} {2} Expected \n", 
                    Utility.ArrayToString(xor[r].input), 
                    Utility.ArrayToString(nn.ComputeNN(xor[r].input), "F0"), 
                    Utility.ArrayToString(xor[r].target)));
            }

            // hold program to 
            Console.ReadLine();
        }

        /// <summary>
        /// Holds information about XOR gates
        /// </summary>
        public struct XOR
        {
            public double[] input;
            public double[] target;

            public XOR(double[] _input, double[] _target)
            {
                input = _input;
                target = _target;
            }
        }
    }
}
