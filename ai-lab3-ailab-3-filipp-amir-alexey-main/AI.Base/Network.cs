using AI.Base.Core;
using AI.Base.LossFunctions;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AI.Base
{
    public class Network
    {
        private Layer[] _layers;
        private LossFunction _lossFunction;

        public Network(Layer[] layers, LossFunction lossFunction)
        {
            _layers = layers;
            _lossFunction = lossFunction;
            InitializeLayers();
        }

        public DataRow Predict(DataRow input)
        {
            var currentLayer = input;
            foreach(var layer in _layers)
            {
                currentLayer = layer.Process(currentLayer);
            }

            if (_lossFunction is CrossEntropyLoss ce)
            {
                currentLayer = ce.SoftMax(currentLayer);
            }

            return currentLayer;
        }
        public List<DataRow> PredictGroup(List<DataRow> input)
        {
            var currentLayer = input;
            foreach (var layer in _layers)
            {
                currentLayer = layer.ProcessGroup(currentLayer);
            }

            if (_lossFunction is CrossEntropyLoss ce)
            {
                currentLayer = ce.SoftMax(currentLayer);
            }

            return currentLayer;
        }

        public void Train(TrainData data, Optimizer optimizer)
        {
            if (data.Input.Count != data.Output.Count)
            {
                throw new ArgumentException("Неправильная база.", nameof(data));
            }

            var predictedValues = PredictGroup(data.Input);
            var expectedValues = data.Output;
            var inputValues = data.Input;

            var layerDelta = new List<DataRow>();
            for (int i = 0; i < data.Input.Count; i++)
            {
                layerDelta.Add(_lossFunction.GetDerivativeFor(predictedValues[i], expectedValues[i]));
            }

            for (int i = _layers.Length - 1; i > 0; i--)
            {
                if (_layers[i] is Core.Activator activator)
                {
                    for (int ld = 0; ld < layerDelta.Count; ld++)
                    {
                        layerDelta[ld] = layerDelta[ld] ^ activator.GetDerivativeFor(_layers[i - 1].IntermediateValues[ld]);
                    }
                }
                else if (_layers[i] is Layers.Linear linearLayer)
                {
                    var weightDelta = new DenseMatrix(layerDelta.FirstOrDefault().Length, _layers[i - 1].IntermediateValues.FirstOrDefault().Length);
                    var biasDelta = new DataRow(layerDelta.FirstOrDefault().Length);
                    for (int ld = 0; ld < layerDelta.Count; ld++)
                    {
                        weightDelta += DataRow.MultiplyColToRow(layerDelta[ld], _layers[i - 1].IntermediateValues[ld]);
                        biasDelta += layerDelta[ld];
                    }

                    weightDelta = weightDelta * (1.0 / layerDelta.Count);
                    biasDelta = biasDelta * (1.0 / layerDelta.Count);

                    //
                    linearLayer.Weights -= optimizer.Optimize(weightDelta);
                    linearLayer.Bias -= optimizer.Optimize(biasDelta);

                    layerDelta = layerDelta.Select(delta =>
                    {
                        var valuesDelta = new DenseMatrix(1, delta.Length, delta.Data) * linearLayer.Weights;
                        return new DataRow(valuesDelta.Row(0).AsArray());
                    }).ToList();
                }
            }

            if (_layers[0] is Layers.Linear firstLayer)
            {
                var weightDelta = new DenseMatrix(layerDelta.FirstOrDefault().Length, inputValues.FirstOrDefault().Length);
                var biasDelta = new DataRow(layerDelta.FirstOrDefault().Length);
                for (int ld = 0; ld < layerDelta.Count; ld++)
                {
                    weightDelta += DataRow.MultiplyColToRow(layerDelta[ld], inputValues[ld]);
                    biasDelta += layerDelta[ld];
                }

                weightDelta = weightDelta * (1.0 / layerDelta.Count);
                biasDelta = biasDelta * (1.0 / layerDelta.Count);

                firstLayer.Weights -= optimizer.Optimize(weightDelta);
                firstLayer.Bias -= optimizer.Optimize(biasDelta);
            }

        }

        private void InitializeLayers()
        {
            foreach(var layer in _layers)
            {
                layer.Initialize();
            }
        }
    }
}
