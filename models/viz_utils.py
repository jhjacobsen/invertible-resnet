VISDOMWINDOWS = {}


def line_plot(viz, title, x, y):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.line(X=[x], Y=[y], win=window, update='append', opts={'title': title})
    else:
        window = viz.line(X=[x], Y=[y], opts={'title': title})
        VISDOMWINDOWS[title] = window


def scatter_plot(viz, title, x):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.scatter(X=x, win=window, update='replace', opts={'title': title})
    else:
        window = viz.scatter(X=x, opts={'title': title})
        VISDOMWINDOWS[title] = window


def images_plot(viz, title, x):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.images(x, win=window, opts={'title': title})
    else:
        window = viz.images(x, opts={'caption': title})
        VISDOMWINDOWS[title] = window