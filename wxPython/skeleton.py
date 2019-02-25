import logging
import logging.config

import wx
import wx.lib.agw.aui as aui

LOGGER = "wxApp"
dictLogConfig = {
  "version":1,
  "handlers":{
    "fileHandler":{
      "class":"logging.FileHandler",
      "formatter":"myFormatter",
      "filename":"test.log"
    },
    "consoleHandler":{
      "class":"logging.StreamHandler",
      "formatter":"myFormatter"
    }
  },        
  "loggers":{
    LOGGER:{
      "handlers":["fileHandler", "consoleHandler"],
      #"level":"INFO",
      "level":"DEBUG",
    }
  },
  "formatters":{
    "myFormatter":{
      "format":"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
  }
}


class CustomConsoleHandler(logging.StreamHandler):
  def __init__(self, textctrl):
    logging.StreamHandler.__init__(self)
    self.textctrl = textctrl
 
 
  def emit(self, record):
    msg = self.format(record) + "\n"
    #self.textctrl.WriteText(msg)
    #self.flush()
    wx.CallAfter(self.textctrl.WriteText, msg)


class TabPanelOne(wx.Panel):
  def __init__(self, parent, name="name"):
    wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
    self.logger = logging.getLogger(LOGGER)
    self.logger.debug(self.__class__.__name__ + '.__init__')

    self.SetName(name)

    self.counter = {}
    txtOne = wx.TextCtrl(self, wx.ID_ANY, "")
    txtOne.SetBackgroundColour(wx.Colour("WHITE"))
    txtTwo = wx.TextCtrl(self, wx.ID_ANY, "")
    txtTwo.SetBackgroundColour(wx.Colour('WHITE'))

    button = wx.Button(self, wx.ID_ANY, label='Test', pos=(10, 10), name="Test")
    button.SetBackgroundColour(wx.Colour('LIGHT GREY'))
    self.counter.update({button.GetName(): 0})

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(txtOne, 0, wx.ALL, 5)
    sizer.Add(txtTwo, 0, wx.ALL, 5)
    sizer.Add(button, 0, wx.ALL, 5)

    button.Bind(wx.EVT_BUTTON, self.OnButton)

    self.SetSizer(sizer)

  def OnButton(self, event):
    o = event.GetEventObject()
    self.counter[o.GetName()] += 1
    self.logger.debug(self.__class__.__name__ + '.OnButton: ' + o.GetName() + " button pressed %d times" % self.counter[o.GetName()])


class MenuPanel(wx.Panel):
  def __init__(self, parent):
    self.logger = logging.getLogger(LOGGER)
    self.logger.debug(self.__class__.__name__ + '.__init__')

    wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY, size=wx.Size(-1, 200))

    sizer = wx.BoxSizer(wx.VERTICAL)
    grid_sizer = wx.FlexGridSizer(0, 2, 0, 0)
    sizer.Add(grid_sizer, 0, wx.ALL, 5)

    self.log_message = wx.TextCtrl(self, wx.ID_ANY, size=wx.Size(980, 180),
                                   style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
    self.log_message.SetBackgroundColour(wx.Colour('GREY'))
    grid_sizer.Add(self.log_message, 1, wx.ALL, 5)

    self.SetSizer(sizer)

    txtHandler = CustomConsoleHandler(self.log_message)
    self.logger.addHandler(txtHandler)


class ListNB(wx.Panel):
  def __init__(self, parent, status_bar):
    self.logger = logging.getLogger(LOGGER)
    self.logger.debug(self.__class__.__name__ + '.__init__')

    wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

    self.status_bar = status_bar

    # create the AuiNotebook instance
    style = aui.AUI_NB_DEFAULT_STYLE 
    style &= ~(aui.AUI_NB_CLOSE_ON_ACTIVE_TAB)
    self.nb = aui.AuiNotebook(self, style=style)
    self.Bind(aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.OnClose)

    # add some pages to the notebook
    self.algorithm = TabPanelOne(self.nb, name="Algorithm")
    self.scale = TabPanelOne(self.nb, name="Scale")
    self.nb.AddPage(self.algorithm, self.algorithm.GetName(), False)
    self.nb.AddPage(self.scale, self.scale.GetName(), False)

    self.list_pages = dict()
    self.list_pages.update({self.algorithm.GetName() : (self.algorithm, 0)})
    self.list_pages.update({self.scale.GetName() : (self.scale, 1)})

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(self.nb, 1, wx.EXPAND)
    self.SetSizer(sizer)


  def OnClose(self, event):
    self.logger.debug(self.__class__.__name__ + '.OnClose')

    event.Veto()
    self.status_bar.SetStatusText("vetoed EVT_AUINOTEBOOK_PAGE_CLOSE")
    


class MainFrame(wx.Frame):
  def __init__(self):
    self.logger = logging.getLogger(LOGGER)
    self.logger.debug(self.__class__.__name__ + '.__init__')

    wx.Frame.__init__(self, None, wx.ID_ANY, "GUI skeleton", size=(1000,800))

    self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

    self.status_bar = self.CreateStatusBar()
    self.status_bar.SetStatusText("Ready")

    self.CreateMenuBar()

    self.Bind(wx.EVT_CLOSE, self.OnClose)

    style = wx.TR_HAS_BUTTONS | wx.TR_HIDE_ROOT
    self.tree = wx.TreeCtrl(self, 1, wx.DefaultPosition, (200,200), style)

    root = self.tree.AddRoot('root')
    user = self.tree.AppendItem(root, 'User Defined')
    self.tree.AppendItem(user, 'Algorithm')
    predefined = self.tree.AppendItem(root, 'Predefined')
    self.tree.AppendItem(predefined, 'Scale')
    self.tree.ExpandAll()
    self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnNodeChanged, self.tree)


    self.menu = MenuPanel(self)
    self.list_panel = ListNB(self, self.status_bar)

    self._mgr = aui.AuiManager(self)
    #self._mgr.SetManagedWindow(self)

    self._mgr.AddPane(self.tree,
                      aui.AuiPaneInfo().Name("Tree")
                                       .Caption("Tree")
                                       .Left()
                                       .Layer(1)
                                       .Position(0)
                                       .CloseButton(False)
                                       .MaximizeButton(False)
                                       .MaxSize((200, 200)))

    self._mgr.AddPane(self.list_panel,
                      aui.AuiPaneInfo().Name("Tabs")
                                       .Caption("Tabs")
                                       .Center()
                                       .Layer(1)
                                       .Position(0)
                                       .CloseButton(False)
                                       .MaximizeButton(True)
                                       .MinSize((-1,400)))
    self._mgr.AddPane(self.menu,
                      aui.AuiPaneInfo().Name("Info panel")
                                       .Caption("Log message")
                                       .Bottom()
                                       .Layer(1)
                                       .Position(0)
                                       .CloseButton(False)
                                       .MaximizeButton(True)
                                       .MaxSize((-1, 100)))

    self._mgr.GetPane("Tabs").dock_proportion = 10
    self._mgr.GetPane("Tree").dock_proportion = 100
    self._mgr.GetPane("Info panel").dock_proportion = 10

    self._mgr.Update()

    self.Show()


  def CreateMenuBar(self):
    self.logger.debug(self.__class__.__name__ + '.CreateMenuBar')

    mb = wx.MenuBar()
    self.SetMenuBar(mb)

    menu = wx.Menu()
    mb.Append(menu, "&File")

    open_file = menu.Append(wx.ID_OPEN, "Open File", "Open some file")
    write_menu = menu.Append(wx.ID_SAVEAS, "Save File", " Save some file")
    exit = menu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

    self.Bind(wx.EVT_MENU, self.OnOpenFile, open_file)
    self.Bind(wx.EVT_MENU, self.OnClose, exit)


  def OnClose(self, event):
    self.logger.debug(self.__class__.__name__ + '.OnClose')

    self._mgr.UnInit()
    self.Destroy()


  def OnOpenFile(self, event):
    self.logger.debug(self.__class__.__name__ + '.OnOpenFile')

    dir_name = ''
    dlg = wx.FileDialog(self, "Choose a text file", dir_name,
                        "", "*.txt", wx.FD_OPEN)
    if dlg.ShowModal() == wx.ID_OK:
      print(dlg.GetFilename())
      print(dlg.GetDirectory())
      import os
      self.status_bar.SetStatusText("Opened " + os.path.join(dlg.GetDirectory(), dlg.GetFilename()))

    dlg.Destroy()


  def OnNodeChanged(self, event):
    self.logger.debug(self.__class__.__name__ + '.OnNodeChanged')

    item = event.GetItem()
    name = self.tree.GetItemText(item)
    try:
      index = self.list_panel.list_pages[name][1]
    except KeyError:
      return
    self.list_panel.nb.SetSelection(index)


if __name__ == "__main__":
  import os.path
  logging.config.dictConfig(dictLogConfig)
  logger = logging.getLogger(LOGGER)
 
  logging.addLevelName(10, 'dbg')
  logging.addLevelName(20, 'inf')
  logging.addLevelName(30, 'war')
  logging.addLevelName(40, 'err')
  logging.addLevelName(50, 'fat')

  logging.info('started')
  base = os.getenv('TMGUI_DIR')
  if base:
    os.chdir(base)
    logging.info('app dir -> %s' % base)
  app = wx.App()
  frame = MainFrame()
  app.MainLoop()
  logging.info('finished')

# eof
