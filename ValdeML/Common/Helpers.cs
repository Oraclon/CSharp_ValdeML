using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ValdeML
{
    public static class Transposer
    {
        public static double[][] TransposeList(double[][] inputs)
        {
            int outter_size = inputs.Length;
            int inner_size = inputs[0].Length;

            double[][] transposed = new double[inner_size][];

            for (int i = 0; i < inner_size; i++)
            {
                double[] test = new double[outter_size];

                Span<double> outterSizeAsSpan = Enumerable.Range(0, outter_size).Select(x=> Convert.ToDouble(x)).ToArray();
                ref var searchSpace = ref MemoryMarshal.GetReference(outterSizeAsSpan);
                for (int j = 0; j < outterSizeAsSpan.Length; j++)
                {
                    var selectedId = (int)Unsafe.Add(ref searchSpace, j);
                    test[j] = inputs[selectedId][i];
                }

                transposed[i] = test;
            }
            return transposed;
        }
    }

    //public static class Batches
    //{
    //    public static MMODEL[][] Get(MMODEL[] dataset, int batchsize)
    //    {
    //        int totalbatches = dataset.Length / batchsize;
    //        MMODEL[][] batches= new MMODEL[totalbatches][];
    //        int bid = 0;
    //        for(int x= 0; x< dataset.Length; x+= batchsize)
    //        {
    //            if (bid.Equals(totalbatches))
    //                continue;
    //            MMODEL[] batch = dataset.Skip(x).Take(batchsize).ToArray();
    //            batches[bid] = batch;
    //            bid++;
    //        }
    //        return batches;
    //    }
    //}
}

