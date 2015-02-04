using System;
using System.Windows.Forms;

namespace CudaMatrix.Editor
{
    public partial class MainForm : Form
    {
        static readonly RandomDialog RandomDialog = new RandomDialog();
        public MainForm()
        {
            InitializeComponent();
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void newCudaEditorForm_Click(object sender, EventArgs e)
        {
            var child = new CudaEditorForm
            {
                MdiParent = this
            };
            child.Show();
        }

        private void executeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as EditorForm;
            if (child == null) return;
            child.Execute();
        }

        private void randomToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as EditorForm;
            if (child == null) return;
            if (RandomDialog.ShowDialog() != DialogResult.OK) return;
            child.Random(RandomDialog.Minimum,RandomDialog.Maximum);
        }

        private void newMpiEditorForm_Click(object sender, EventArgs e)
        {
            var child = new MpiEditorForm
            {
                MdiParent = this
            };
            child.Show();
        }
    }
}