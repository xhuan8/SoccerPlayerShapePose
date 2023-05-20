using LCommon.Vision;
using LCommon.Vision.Diagnostics;
using LControl;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Soccer
{
    /// <summary>
    /// PlayerCropWindow.xaml 的交互逻辑
    /// </summary>
    public partial class PlayerCropWindow : Window
    {
        private List<string> images = new List<string>();
        private List<string> data = new List<string>();
        private List<string> indexImages = new List<string>();
        private List<string> playerImages = new List<string>();
        private int current = -1;
        private int currentPlayer = -1;
        private LImage currentImage;
        private LImage currentPlayerImage;
        private const int borderX = 15;
        private const int borderY = 10;
        private List<OverlayText> overlayText = new List<OverlayText>();
        private string imageFolder;
        private string dataFolder;
        private bool isMatchIndex;

        public PlayerCropWindow()
        {
            InitializeComponent();
        }

        public void Initialize(string imageFolder, string dataFolder, bool isMatchIndex = true)
        {
            this.imageFolder = imageFolder;
            this.dataFolder = dataFolder;
            this.isMatchIndex = isMatchIndex;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            string selectedGame = "";

            bool ignore = string.IsNullOrEmpty(selectedGame);
            foreach (string game in Directory.GetDirectories(imageFolder))
            {
                if (ignore)
                {
                    if (game.Contains(selectedGame))
                        ignore = false;
                    else
                        continue;
                }

                string gameData = System.IO.Path.Combine(dataFolder, System.IO.Path.GetFileName(game));
                foreach (string scene in Directory.GetDirectories(game))
                {
                    string sceneData = System.IO.Path.Combine(gameData, System.IO.Path.GetFileName(scene));
                    images.Add(System.IO.Path.Combine(scene, "broad.png"));
                    data.Add(System.IO.Path.Combine(sceneData, "boxes.xml"));
                    indexImages.Add(System.IO.Path.Combine(scene, "1"));
                }
            }
        }

        private void btnPrevious_Click(object sender, RoutedEventArgs e)
        {
            current -= 1;
            UpdateDisplay();
        }

        private void btnNext_Click(object sender, RoutedEventArgs e)
        {
            current += 1;
            UpdateDisplay();
        }

        private void UpdateDisplay()
        {
            if (current >= 0 && current < images.Count)
            {
                if (currentImage != null)
                    currentImage.Dispose();
                currentImage = new LImage();
                currentImage.CreateFromFile(images[current]);

                imageDisplay.Image = currentImage;
                txtImageFilename.Text = images[current];

                string json = File.ReadAllText(data[current]);
                var boxes = JsonConvert.DeserializeObject(json) as JArray;
                List<BaseROI> rois = new List<BaseROI>();
                overlayText = new List<OverlayText>();

                JArray indexes = null;
                if (File.Exists(data[current].Replace("boxes.xml", "index.xml")))
                {
                    string indexJson = File.ReadAllText(data[current].Replace("boxes.xml", "index.xml"));
                    indexes = JsonConvert.DeserializeObject(indexJson) as JArray;
                }
                for (int i = 0; i < boxes.Count; i++)
                {
                    var box = boxes[i];
                    int x = (int)box[0], y = (int)box[1], x1 = (int)box[2], y1 = (int)box[3];
                    x -= borderX;
                    if (x < 0)
                        x = 0;
                    y -= borderY;
                    if (y < 0)
                        y = 0;
                    x1 += borderX;
                    if (x1 >= currentImage.Width)
                        x1 = currentImage.Width - 1;
                    y1 += borderY;
                    if (y1 >= currentImage.Height)
                        y1 = currentImage.Height - 1;
                    RectangleROI rect = new RectangleROI(x, y, x1 - x, y1 - y);
                    rois.Add(rect);

                    string ii = "";
                    if (indexes != null && indexes.Count > i)
                        ii = (string)indexes[i];
                    overlayText.Add(new OverlayText(x, y - 10 >= 0 ? y - 10 : y, ii));
                }
                imageDisplay.ROIList = rois;

                playerImages = new List<string>();

                if (Directory.Exists(indexImages[current]))
                {
                    foreach (string player in Directory.GetFiles(indexImages[current]))
                    {
                        playerImages.Add(player);
                    }
                }

                currentPlayer = -1;
                btnNextDisplay_Click(btnNext, null);
                RebuildOverlay();
            }
            else
            {
                imageDisplay.Image = null;
                txtImageFilename.Text = string.Empty;
                imageDisplayIndex.Image = null;
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (currentImage != null)
                currentImage.Dispose();

            if (currentPlayerImage != null)
                currentPlayerImage.Dispose();
        }

        private void btnDelete_Click(object sender, RoutedEventArgs e)
        {
            if (imageDisplay.SelectedRoi != null)
            {
                int index = imageDisplay.ROIList.IndexOf(imageDisplay.SelectedRoi);
                if (index < 0)
                    return;
                imageDisplay.ROIList.Remove(imageDisplay.SelectedRoi);
                imageDisplay.ROIList = new List<BaseROI>(imageDisplay.ROIList);
                imageDisplay.SelectedRoi = null;
                overlayText.RemoveAt(index);
                RebuildOverlay();
            }
        }

        private void btnAdd_Click(object sender, RoutedEventArgs e)
        {
            RectangleROI roi = new RectangleROI(0, 0, 100, 100);
            imageDisplay.ROIList.Add(roi);
            imageDisplay.ROIList = new List<BaseROI>(imageDisplay.ROIList);

            overlayText.Add(new OverlayText(0, 0, ""));
        }

        private void btnSave_Click(object sender, RoutedEventArgs e)
        {
            string filename = data[current];

            List<List<int>> lists = new List<List<int>>();
            foreach (BaseROI roi in imageDisplay.ROIList)
            {
                List<int> list = new List<int>();
                RectangleROI rect = roi as RectangleROI;
                list.Add(rect.Left);
                list.Add(rect.Top);
                list.Add(rect.Right);
                list.Add(rect.Bottom);
                lists.Add(list);
            }
            string json = JsonConvert.SerializeObject(lists);
            File.WriteAllText(filename, json);

            if (isMatchIndex)
            {
                bool finish = true;
                foreach (OverlayText item in overlayText)
                    if (string.IsNullOrEmpty(item.Text))
                        finish = false;
                if (!finish)
                {
                    MessageBox.Show("没有完成");
                    return;
                }

                string indexFilename = filename.Replace("boxes.xml", "index.xml");
                List<string> indexLists = new List<string>();
                foreach (OverlayText item in overlayText)
                {
                    indexLists.Add(item.Text);
                }
                json = JsonConvert.SerializeObject(indexLists);
                File.WriteAllText(indexFilename, json);
            }
        }

        private void btnMatch_Click(object sender, RoutedEventArgs e)
        {
            string filename = playerImages[currentPlayer];
            string index = System.IO.Path.GetFileNameWithoutExtension(filename);
            index = index.Substring(index.IndexOf('_')+1);
            int id = int.Parse(index) + 1;

            RectangleROI selectedRoi = imageDisplay.SelectedRoi as RectangleROI;
            int roiIndex = imageDisplay.ROIList.IndexOf(selectedRoi);
            overlayText[roiIndex].Text = id.ToString();

            RebuildOverlay();
        }

        private void RebuildOverlay()
        {
            Overlay overlay = new Overlay();
            for (int i = 0; i < overlayText.Count; i++)
            {
                overlayText[i].Pos = new Point(imageDisplay.ROIList[i].Left, imageDisplay.ROIList[i].Top);
                overlay.AddObject(overlayText[i]);
            }
            imageDisplay.Overlay = overlay;
        }

        private void btnNextDisplay_Click(object sender, RoutedEventArgs e)
        {
            currentPlayer += 1;
            if (currentPlayer >= playerImages.Count)
                currentPlayer = 0;

            if (currentPlayerImage != null)
                currentPlayerImage.Dispose();
            if (currentPlayer < playerImages.Count)
            {
                currentPlayerImage = new LImage();
                currentPlayerImage.CreateFromFile(playerImages[currentPlayer]);

                imageDisplayIndex.Image = currentPlayerImage;
            }

            lblIndex.Text = (currentPlayer + 1).ToString();
        }

        private void btnPreDisplay_Click(object sender, RoutedEventArgs e)
        {
            currentPlayer -= 1;
            if (currentPlayer < 0)
                currentPlayer = playerImages.Count - 1;

            if (currentPlayerImage != null)
                currentPlayerImage.Dispose();
            currentPlayerImage = new LImage();
            currentPlayerImage.CreateFromFile(playerImages[currentPlayer]);

            imageDisplayIndex.Image = currentPlayerImage;
            lblIndex.Text = (currentPlayer + 1).ToString();
        }
    }
}
