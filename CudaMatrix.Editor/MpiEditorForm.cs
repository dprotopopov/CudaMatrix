using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

namespace CudaMatrix.Editor
{
    public partial class MpiEditorForm : Form, EditorForm
    {
        public enum OpCodes
        {
            Unknown,
            Add,
            Sub,
            Mul,
            Mtv,
            Det,
            Inv,
            Mov,
            One,
            Nil,
            Rot,
        }

        private static readonly Random Rnd = new Random();

        private readonly MatrixIO _dataGridViewMatrixA = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        private readonly MatrixIO _dataGridViewMatrixB = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        private readonly MatrixIO _dataGridViewMatrixC = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewMatrix",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        public MpiEditorForm()
        {
            InitializeComponent();
            tabPageA.Controls.Add(_dataGridViewMatrixA);
            tabPageB.Controls.Add(_dataGridViewMatrixB);
            tabPageC.Controls.Add(_dataGridViewMatrixC);
            numericUpDownHeightA.ValueChanged += ValueChanged;
            numericUpDownHeightB.ValueChanged += ValueChanged;
            numericUpDownWidthA.ValueChanged += ValueChanged;
            numericUpDownWidthB.ValueChanged += ValueChanged;
            comboBoxOpCode.SelectedIndexChanged += ValueChanged;
        }

        public int HeightA
        {
            get { return Convert.ToInt32(numericUpDownHeightA.Value); }
            set { numericUpDownHeightA.Value = value; }
        }

        public int HeightB
        {
            get { return Convert.ToInt32(numericUpDownHeightB.Value); }
            set { numericUpDownHeightB.Value = value; }
        }

        public int HeightC
        {
            get { return Convert.ToInt32(numericUpDownHeightC.Value); }
            set { numericUpDownHeightC.Value = value; }
        }

        public int WidthA
        {
            get { return Convert.ToInt32(numericUpDownWidthA.Value); }
            set { numericUpDownWidthA.Value = value; }
        }

        public int WidthB
        {
            get { return Convert.ToInt32(numericUpDownWidthB.Value); }
            set { numericUpDownWidthB.Value = value; }
        }

        public int WidthC
        {
            get { return Convert.ToInt32(numericUpDownWidthC.Value); }
            set { numericUpDownWidthC.Value = value; }
        }


        public string OpCode
        {
            get { return comboBoxOpCode.Text; }
        }

        public OpCodes OpCodeId
        {
            get
            {
                string opcode = OpCode;
                if (String.CompareOrdinal(opcode, "add") == 0) return OpCodes.Add;
                if (String.CompareOrdinal(opcode, "sub") == 0) return OpCodes.Sub;
                if (String.CompareOrdinal(opcode, "mul") == 0) return OpCodes.Mul;
                if (String.CompareOrdinal(opcode, "mtv") == 0) return OpCodes.Mtv;
                if (String.CompareOrdinal(opcode, "rot") == 0) return OpCodes.Rot;
                if (String.CompareOrdinal(opcode, "inv") == 0) return OpCodes.Inv;
                if (String.CompareOrdinal(opcode, "det") == 0) return OpCodes.Det;
                if (String.CompareOrdinal(opcode, "mov") == 0) return OpCodes.Mov;
                if (String.CompareOrdinal(opcode, "one") == 0) return OpCodes.One;
                if (String.CompareOrdinal(opcode, "nil") == 0) return OpCodes.Nil;
                return OpCodes.Unknown;
            }
        }

        private int NumberOfProcess
        {
            get { return (int) numericUpDownNumberOfProcess.Value; }
        }

        private string Script
        {
            get { return textBoxScript.Text; }
            set { textBoxScript.Text = value; }
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            string fileNameA = Path.GetTempPath() + "MATRIX_A.bin";
            string fileNameB = Path.GetTempPath() + "MATRIX_B.bin";
            string fileNameC = Path.GetTempPath() + "MATRIX_C.bin";
            var sb = new StringBuilder();
            sb.Append(" " + OpCode);
            switch (OpCodeId)
            {
                case OpCodes.Add:
                case OpCodes.Sub:
                case OpCodes.Mul:
                case OpCodes.Mtv:
                    sb.Append(" \"" + fileNameA + "\"");
                    sb.Append(" \"" + fileNameB + "\"");
                    sb.Append(" \"" + fileNameC + "\"");
                    break;
                case OpCodes.Rot:
                case OpCodes.Inv:
                case OpCodes.Mov:
                case OpCodes.Det:
                    sb.Append(" \"" + fileNameA + "\"");
                    sb.Append(" \"" + fileNameB + "\"");
                    break;
                case OpCodes.One:
                    sb.Append(" " + HeightA);
                    sb.Append(" \"" + fileNameA + "\"");
                    break;
                case OpCodes.Nil:
                    sb.Append(" " + HeightA);
                    sb.Append(" " + WidthA);
                    sb.Append(" \"" + fileNameA + "\"");
                    break;
            }

            Script = sb.ToString();

            switch (OpCodeId)
            {
                case OpCodes.Add:
                case OpCodes.Sub:
                    HeightB = HeightA;
                    HeightC = HeightA;
                    WidthB = WidthA;
                    WidthC = WidthA;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = false;
                    numericUpDownHeightB.ReadOnly = true;
                    numericUpDownWidthB.ReadOnly = true;
                    numericUpDownHeightC.ReadOnly = true;
                    numericUpDownWidthC.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = true;
                    numericUpDownWidthC.Visible = true;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = true;
                    break;
                case OpCodes.Mul:
                    HeightC = HeightA;
                    WidthA = HeightB;
                    WidthC = WidthB;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = true;
                    numericUpDownHeightB.ReadOnly = false;
                    numericUpDownWidthB.ReadOnly = false;
                    numericUpDownHeightC.ReadOnly = true;
                    numericUpDownWidthC.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = true;
                    numericUpDownWidthC.Visible = true;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = true;
                    break;
                case OpCodes.Inv:
                    WidthA = HeightA;
                    WidthB = HeightA;
                    HeightB = HeightA;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = true;
                    numericUpDownHeightB.ReadOnly = true;
                    numericUpDownWidthB.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.Mov:
                    HeightB = HeightA;
                    WidthB = WidthA;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = false;
                    numericUpDownHeightB.ReadOnly = true;
                    numericUpDownWidthB.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.Rot:
                    HeightB = WidthA;
                    WidthB = HeightA;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = false;
                    numericUpDownHeightB.ReadOnly = true;
                    numericUpDownWidthB.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.One:
                    WidthA = HeightA;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = false;
                    numericUpDownWidthB.Visible = false;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = false;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.Nil:

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = false;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = false;
                    numericUpDownWidthB.Visible = false;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = false;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.Det:
                    WidthA = HeightA;
                    WidthB = HeightB = 1;

                    numericUpDownHeightA.ReadOnly = false;
                    numericUpDownWidthA.ReadOnly = true;
                    numericUpDownHeightB.ReadOnly = true;
                    numericUpDownWidthB.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = false;
                    numericUpDownWidthC.Visible = false;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = false;
                    break;
                case OpCodes.Mtv:
                    WidthA = HeightA = 1;
                    HeightC = HeightB;
                    WidthC = WidthB;

                    numericUpDownHeightA.ReadOnly = true;
                    numericUpDownWidthA.ReadOnly = true;
                    numericUpDownHeightB.ReadOnly = false;
                    numericUpDownWidthB.ReadOnly = false;
                    numericUpDownHeightC.ReadOnly = true;
                    numericUpDownWidthC.ReadOnly = true;

                    numericUpDownHeightA.Visible = true;
                    numericUpDownWidthA.Visible = true;
                    numericUpDownHeightB.Visible = true;
                    numericUpDownWidthB.Visible = true;
                    numericUpDownHeightC.Visible = true;
                    numericUpDownWidthC.Visible = true;

                    tabPageA.Visible = true;
                    tabPageB.Visible = true;
                    tabPageC.Visible = true;
                    break;
            }

            _dataGridViewMatrixA.ResizeOurself(HeightA, WidthA);
            _dataGridViewMatrixB.ResizeOurself(HeightB, WidthB);
            _dataGridViewMatrixC.ResizeOurself(HeightC, WidthC);
        }

        public static string[,] ReadMpiMatrix(string fileName)
        {
            using (var reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
            {
                int size = Marshal.SizeOf(typeof (MpiMatrixHeader));
                var arr = new byte[size];
                reader.Read(arr, 0, size);

                IntPtr ptr = Marshal.AllocHGlobal(size);
                Marshal.Copy(arr, 0, ptr, size);

                var header = (MpiMatrixHeader)Marshal.PtrToStructure(ptr, typeof(MpiMatrixHeader));
                Marshal.FreeHGlobal(ptr);

                var matrix = new string[header.height, header.width];
                for (int i = 0; i < matrix.GetLength(0); i++)
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        matrix[i, j] = reader.ReadDouble().ToString(CultureInfo.InvariantCulture);
                reader.Close();
                return matrix;
            }
        }

        public static void WriteMpiMatrix(string fileName, string[,] matrix)
        {
            using (var writer = new BinaryWriter(File.Open(fileName, FileMode.Create)))
            {
                byte[] bytes = {(byte) 'M', (byte) 'P', (byte) 'I', (byte) 'M'};
                var header = new MpiMatrixHeader
                {
                    fourCC = BitConverter.ToUInt32(bytes, 0),
                    dataType = (uint) MPI_Datatype.MPI_DOUBLE,
                    height = (uint) matrix.GetLength(0),
                    width = (uint) matrix.GetLength(1),
                    offset = (uint) Marshal.SizeOf(typeof (MpiMatrixHeader))
                };

                int size = Marshal.SizeOf(typeof (MpiMatrixHeader));
                IntPtr ptr = Marshal.AllocHGlobal(size);

                // Copy the struct to unmanaged memory.
                Marshal.StructureToPtr(header, ptr, true);

                var arr = new byte[size];
                Marshal.Copy(ptr, arr, 0, size);
                Marshal.FreeHGlobal(ptr);

                writer.Write(arr);
                NumberFormatInfo provider = new NumberFormatInfo();
                provider.NumberDecimalSeparator = ".";
                for (int i = 0; i < matrix.GetLength(0); i++)
                    for (int j = 0; j < matrix.GetLength(1); j++)
                    {
                        double value = Convert.ToDouble(matrix[i, j], provider);
                        writer.Write(value);
                    }
                writer.Close();
            }
        }

        public async void Execute()
        {
            string fileNameLog = Path.GetTempPath() + "MATRIX.log";
            string fileNameA = Path.GetTempPath() + "MATRIX_A.bin";
            string fileNameB = Path.GetTempPath() + "MATRIX_B.bin";
            string fileNameC = Path.GetTempPath() + "MATRIX_C.bin";

            switch (OpCodeId)
            {
                case OpCodes.Add:
                case OpCodes.Sub:
                case OpCodes.Mul:
                case OpCodes.Mtv:
                    WriteMpiMatrix(fileNameA, _dataGridViewMatrixA.TheData);
                    WriteMpiMatrix(fileNameB, _dataGridViewMatrixB.TheData);
                    break;
                case OpCodes.Rot:
                case OpCodes.Inv:
                case OpCodes.Mov:
                case OpCodes.Det:
                    WriteMpiMatrix(fileNameA, _dataGridViewMatrixA.TheData);
                    break;
                case OpCodes.One:
                case OpCodes.Nil:
                    break;
            }
            string command = string.Format("/C mpiexec.exe -n {0} mpimatrix.exe {1} > {2}",
                NumberOfProcess,
                Script,
                fileNameLog);
            Debug.WriteLine(command);

            DateTime start = DateTime.Now;
            Process process = Process.Start("cmd", command);
            //MessageBox.Show(command);
            if (process == null) return;
            process.WaitForExit();

            DateTime end = DateTime.Now;

            switch (OpCodeId)
            {
                case OpCodes.Add:
                case OpCodes.Sub:
                case OpCodes.Mul:
                case OpCodes.Mtv:
                    _dataGridViewMatrixC.TheData = ReadMpiMatrix(fileNameC);
                    break;
                case OpCodes.Rot:
                case OpCodes.Inv:
                case OpCodes.Mov:
                case OpCodes.Det:
                    _dataGridViewMatrixB.TheData = ReadMpiMatrix(fileNameB);
                    break;
                case OpCodes.One:
                case OpCodes.Nil:
                    _dataGridViewMatrixA.TheData = ReadMpiMatrix(fileNameA);
                    break;
            }

            using (var reader = new StreamReader(File.Open(fileNameLog, FileMode.Open)))
            {
                textBoxLog.Text = await reader.ReadToEndAsync();
            }

            var timeSpan = new TimeSpan(end.Ticks - start.Ticks);
            MessageBox.Show(timeSpan.ToString());
        }

        public void Random(double minimum, double maximum)
        {
            var matrix = new string[HeightA, WidthA];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] =
                        Convert.ToInt16(minimum + (maximum - minimum)*Rnd.NextDouble())
                            .ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixA.TheData = matrix;
            matrix = new string[HeightB, WidthB];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] =
                        Convert.ToInt16(minimum + (maximum - minimum)*Rnd.NextDouble())
                            .ToString(CultureInfo.InvariantCulture);
                }
            _dataGridViewMatrixB.TheData = matrix;
        }

        private enum MPI_Datatype

        {
            MPI_CHAR = 0x4c000101,
            MPI_SIGNED_CHAR = 0x4c000118,
            MPI_UNSIGNED_CHAR = 0x4c000102,
            MPI_BYTE = 0x4c00010d,
            MPI_WCHAR = 0x4c00020e,
            MPI_SHORT = 0x4c000203,
            MPI_UNSIGNED_SHORT = 0x4c000204,
            MPI_INT = 0x4c000405,
            MPI_UNSIGNED = 0x4c000406,
            MPI_LONG = 0x4c000407,
            MPI_UNSIGNED_LONG = 0x4c000408,
            MPI_FLOAT = 0x4c00040a,
            MPI_DOUBLE = 0x4c00080b,
            MPI_LONG_DOUBLE = 0x4c00080c,
            MPI_UNSIGNED_LONG_LONG = 0x4c000819,
            MPI_LONG_LONG = 0x4c000809
        };

        [StructLayout(LayoutKind.Explicit)]
        private struct MpiMatrixHeader
        {
            [FieldOffset(0)] public UInt32 fourCC; // Идентификатор формата файла http://en.wikipedia.org/wiki/FourCC
            [FieldOffset(4)] public UInt32 dataType; // Тип данных в массиве
            [FieldOffset(8)] public UInt32 height; // Количество строк
            [FieldOffset(12)] public UInt32 width; // Количество столбцов
            [FieldOffset(16)] public UInt64 offset; // Позиция данных в файле
        };
    }
}