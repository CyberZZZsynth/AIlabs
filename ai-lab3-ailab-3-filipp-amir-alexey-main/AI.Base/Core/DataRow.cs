using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Text;

namespace AI.Base.Core
{
    public class DataRow
    {
        public int Length => Data.Length;

        public double[] Data { get; }

        public DataRow(double[] data)
        {
            Data = data;
        }

        public DataRow(int row)
        {
            Data = new double[row];
            for (int i = 0; i < row; i++)
            {
                Data[i] = 0;
            }
        }

        public static DataRow CreateRandom(int rows, ContinuousUniform distribution)
        {
            return new DataRow(Generate.Random(rows, distribution));     
        }

        public static DenseMatrix MultiplyColToRow(DataRow dataRow1, DataRow dataRow2)
        {
            var res = new DenseMatrix(dataRow1.Length, dataRow2.Length);
            for (int i = 0; i < dataRow1.Length; i++)
            {
                for (int j = 0; j < dataRow2.Length; j++)
                {
                    res[i, j] = dataRow1[i] * dataRow2[j];
                }
            }

            return res;
        }

        public double this[int index]
        {
            get => Data[index];
            set => Data[index] = value;
        }

        public override string ToString()
        {
            var buffer = new StringBuilder();
            buffer.Append("[ ");
            foreach(var element in Data)
            {
                buffer.Append(string.Format("{0:0.00;-0.00}", element)).Append(" ");
            }
            buffer.Append("]");
            return buffer.ToString();
        }

        /// <summary>
        /// Поэлементное умножение
        /// </summary>
        public static DataRow operator ^(DataRow a, DataRow b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Разная длина строк");
            }
            
            var res = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] * b[i];
            }

            return new DataRow(res);
        }

        public static DataRow operator -(DataRow a, DataRow b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Разная длина строк");
            }

            var res = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] - b[i];
            }

            return new DataRow(res);
        }

        public static DataRow operator *(DataRow a, double b)
        {
            var res = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] * b;
            }

            return new DataRow(res);
        }

        public static DataRow operator *(double b, DataRow a)
        {
            var res = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] * b;
            }

            return new DataRow(res);
        }
        public static DataRow operator +(DataRow a, DataRow b)
        {
            var res = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] + b[i];
            }

            return new DataRow(res);
        }
    }
}
