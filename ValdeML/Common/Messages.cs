using System;
namespace ValdeML
{
	public static class MLMessages
	{
        #region Info

        #endregion

        #region Exceptions
        public const string NA0001 = "[NA0001]: Demo builder supports only [MinMax] [Mean] [ZScore] scaler.";
        public const string NA0002 = "[NA0002]: Undefined Learning Rate.";
        public const string NA0003 = "[NA0003]: Undefined Model.";
        #endregion

        #region Suggestions
        public const string SUG0001 = "[SUG0001]: Seems like there is ONE LAYER with ONE NODE. Create a new [Node] for this kind of prediction.";
        #endregion
    }
}

