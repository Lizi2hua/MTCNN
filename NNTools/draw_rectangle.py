
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def draw_rect(img_path,label_data):
    '''
    :param img_path: 图片的路径，建议绝对路径
    :param label_data: 数据的格式最少的二维的，未考虑[x,x,x,x,x]的数据的情况，有bug
    :return:
    '''
#注意图片的坐标是以左上角为原点的，画布也应该以左上角为原点
    img=plt.imread(img_path)
    fig,ax=plt.subplots(1)
    ax.imshow(img)

    for rect_data in label_data:
        top=(rect_data[1],rect_data[2])
        w=rect_data[3]-rect_data[1]
        h=rect_data[4]-rect_data[2]
        rect=patches.Rectangle(top,w,h,linewidth=1,edgecolor='r',fill=False)
        plt.annotate("%.2f"%rect_data[0],top)
        ax.add_patch(rect)
    plt.show()
