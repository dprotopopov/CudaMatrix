using System.Windows.Forms;

namespace CudaMatrix.Editor
{
    public interface EditorForm
    {
        void Random(double minimum, double maximum);
        void Execute();
    }
}