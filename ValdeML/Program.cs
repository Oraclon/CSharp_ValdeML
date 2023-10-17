namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {

            Model model = new Model(Errors.LogLoss);
            model.Learning = .2;

            Dataset dataSet = new Dataset();
            dataSet.BuildDemo(300000, 512, 2, Scaler.ZScore, true);

            MLController mlc = new MLController();
            mlc.AddLayer(Activation.Tanh, 4);
            mlc.AddLayer(Activation.Tanh, 4);
            mlc.AddLayer(Activation.Sigmoid, 1);

            mlc.BuildLayers();
            mlc.StartTraining(model, dataSet);

            var ttt = model.Epochs;
        }
    }
}
