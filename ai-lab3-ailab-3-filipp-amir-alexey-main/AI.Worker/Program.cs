using AI.Base;
using AI.Base.Core;
using AI.Base.Optimizers;
using AI.Base.Train;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace AI.Worker
{
    class Program
    {
        public static double[] GetGrayness(string path)
        {
            Bitmap bmp = new Bitmap(path);
            int w = bmp.Width;
            int h = bmp.Height;
            double[] arr = new double[w * h];

            for (int i = 0; i < w * h; i++)
            {
                int x = i % w;
                int y = i / w;
                var c = bmp.GetPixel(x, y);
                double grayness = (c.R + c.G + c.B) / (3.0 * 255.0);
                arr[i] = grayness;
            }

            return arr;
        }

        public static List<(double[] output, double[] layer, string fileName)> GetTopGrayness(string path, int size, int firstIndex = 0)
        {
            var result = new List<(double[], double[], string)>();

            if (Directory.Exists(path))
            {
                string[] files = Directory.GetFiles(path);

                if (files.Length > 0)
                {
                    for (int i = firstIndex; i < Math.Min(files.Length, firstIndex + size); i++)
                    {
                        string fileName = Path.GetFileName(files[i]);
                        var grayness = GetGrayness($"{path}\\{fileName}");
                        result.Add((ExtractNumber(fileName), grayness, fileName));
                    }
                }
            }

            return result;
        }

        public static double[] ExtractNumber(string s)
        {
            int firstDotIndex = s.IndexOf('.');
            int secondDotIndex = s.IndexOf('.', firstDotIndex + 1);
            string numberString = s.Substring(firstDotIndex + 1, secondDotIndex - firstDotIndex - 1);

            int number = int.Parse(numberString);

            var res = new double[10];

            for (int i = 0; i < 10; i++)
            {
                res[i] = 0;
            }
            res[number] = 1;

            return res;
        }

        static void Main(string[] args)
        {/*
            Console.WriteLine("-= TRAINING NETWORK =-");
            var network = new Network(new Layer[] {
                new Base.Layers.Linear { InputSize = 3, OutputSize = 5 },
                new Base.Layers.Linear { InputSize = 5, OutputSize = 10 },
                new Base.Layers.Linear { InputSize = 10, OutputSize = 5 },
            }, new Base.Activators.HyperTan());
            var input = new Base.Core.DataRow(new double[] { 0.5, 0.3, 0.8 });
            Console.WriteLine(input);
            var output = network.Predict(input);
            Console.WriteLine(output);

            var trainData = new Base.Core.TrainData[] {
                new Base.Core.TrainData() { Input = input, Output = output }
            };
            network.Train(trainData);

            var n = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            foreach(var b in Base.Train.Helpers.Batching(n, 3))
            {
                foreach(var e in b)
                {
                    Console.Write(e+" ");
                }
                Console.WriteLine();
            }*/

            var network = new Network(new Layer[]
            {
                new Base.Activators.HyperTan {InputSize = 784, OutputSize = 784},
                new Base.Layers.Linear { InputSize = 784, OutputSize = 4},
                new Base.Activators.ReLU { InputSize = 4, OutputSize = 4},
                new Base.Layers.Linear { InputSize = 4, OutputSize = 20},
                new Base.Activators.HyperTan { InputSize = 20, OutputSize = 20},
                new Base.Layers.Linear { InputSize = 20, OutputSize = 10},
            }, new Base.LossFunctions.CrossEntropyLoss());

            var batchsize = 50;
            var count = 50000;

            var testRawData = GetTopGrayness(@"D:\AI.Framework — копия\AI.Worker\all\all", count, count);

            for (int i = 0; i < count; i += batchsize)
            {
                var trainRawData = GetTopGrayness(@"D:\AI.Framework — копия\AI.Worker\all\all", batchsize, i);
                var input = trainRawData.Select(el => new DataRow(el.layer)).ToList();
                var output = trainRawData.Select(el => new DataRow(el.output)).ToList();

                network.Train(new TrainData() { Input = input, Output = output }, new NesterovOptimizer(0.0001));

                Console.WriteLine(1.0 * i / count);
            }
            Console.WriteLine("Trained");
            var rand = new Random();
            while (true)
            {
                var inp = int.Parse(Console.ReadLine());
                var nums = testRawData.Where(el => el.output[inp] >= 0.8).ToList();

                var ind = rand.Next(nums.Count);

                Console.WriteLine(network.Predict(new DataRow(nums[ind].layer)));
                Console.WriteLine(nums[ind].fileName);
            }
        }
    }
}
