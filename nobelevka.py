import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from bs4 import BeautifulSoup
import plotly.express as px
import seaborn as sns
import squarify
import pycountry
import geopandas
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
from celluloid import Camera

st.title("Nobelevo4ka")

page = st.sidebar.selectbox('Select a section:',('Graphics','Search'))

st.image('1.jpg')
"В этой проге я постараюсь визаулизировать данные по Нобелевской премии 1901-2019. " \
"Данные взяты с Kagle (https://www.kaggle.com/imdevskp/nobel-prize)"
"В датасете содержится информация о литературе/медицине/премии мира/физике/химии и экономике (основана в 1969)"
"Стоит также упомянуть, что в период 1940-1942 Нобелевская премия не вручалась(("




desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

with st.echo(code_location='below'):
    data = pd.read_csv('complete.csv', delimiter=',').sort_values("awardYear").reset_index()
    del data["index"]


with st.echo(code_location='below'):
    data["gender"] = data["gender"].fillna("Organization")
    data_year = pd.DataFrame({"Year": range(1901, 1940), "male": 0, "female": 0, "Organization": 0})
    data_year2 = pd.DataFrame({"Year": range(1943, 2020), "male": 0, "female": 0, "Organization": 0})
    data_year = pd.concat([data_year, data_year2], ignore_index=True)
    for j in ["male", "female", "Organization"]:
        for i in range(1901, 1939):
            data_year.loc[i - 1901][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))
        for i in range(1943, 2020):
            data_year.loc[i - 1904][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))


with st.echo(code_location='below'):
    number = data_year.sum()[1:4].to_list()
    genders = ["male", "female", "Organization"]
    a, b = plt.subplots()

    ### FROM https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct"
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct
    ### END FROM https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct

    b.pie(number, (0, 0, 0.1), genders, ['#60b3ff', '#ff9999', '#99ff99'], autopct=make_autopct(number),
          shadow=True, startangle=11)
    b.axis('equal')
    plt.tight_layout()
    plt.legend(title='Гендерное равенство?! Ну типа', bbox_to_anchor=(1, 1), loc='upper center')

    '''Получился нелепый мужской пакмэн, проглотивший остальных. Действительно, пока что нет никаких новостей,
    с куммулятивным гендерным распределением всё и так был понятно.'''

st.pyplot(plt)

with st.echo(code_location='below'):
    b = ["Chemistry", "Literature", "Physiology or Medicine", "Peace", "Physics", "Economic Sciences"]
    subj = pd.DataFrame({"Category": b, "Total": 0, "Female": 0})
    for j in b:
        data_year[j] = 0
        for i in data_year["Year"]:
            data_year.loc[i - 1901 - 3 * int(i / 1943)][j] = int(
                data[data["awardYear"] == int(i)]["category"].to_list().count(j))
        subj.iloc[b.index(j), 1] = data_year[j].sum()
        subj.iloc[b.index(j), 2] = data[data["category"] == j]["gender"].to_list().count("female")

with st.echo(code_location='below'):
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(4, 4))
    subjects = subj.sort_values("Total", ascending=False)
    sns.barplot(x="Total", y="Category", data=subjects, label="Total", color="#60b3ff")
    sns.set_color_codes("muted")
    sns.barplot(x="Female", y="Category", data=subjects, label="Female", color="#ff9999")
    ax.legend(ncol=1, loc="lower right", frameon=True)
    ax.set(xlim=(0, 250), ylabel="", xlabel="Number of prizes")
    plt.title("Female distribution per categories")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(sns.despine(left=True, bottom=True))
"Как мы видим, процентные соотношения женщин в каждой из категорий очень разнятся. Наибольшую долю они составляют " \
    "в премии Мира и премии по литературе, откуда можно сделать вывод о том, что женщины лучше преуспевают в гуманитарных науках (литература)" \
    "и в социальной активности/иницитивности (премия Мира), нежели в естественных науках ('преуспевают' в данном контексте относится именно" \
    "к Нобелевской премии)"

"Хмммммм. А вы заметили, что во всех категориях количество врученных премий значительно превышает временной промежуток, на протяжении " \
"которого эти премии присуждались (раз в год)??? Омагадддд, получается, одну премию могут получать сразу несколько человек!!!"

with st.echo(code_location='below'):
    y = data[["awardYear", "category"]].copy()
    y["1"] = 1
    y = y.groupby(["awardYear", "category"]).sum("1")
    x = data_year[["Year", "Chemistry", "Literature", "Physiology or Medicine", "Peace",
                   "Physics", "Economic Sciences"]].pivot_table(["Chemistry", "Literature",
                                                                 "Physiology or Medicine", "Peace", "Physics",
                                                                 "Economic Sciences"], "Year")
    x = pd.pivot_table(x, values=["Chemistry", "Literature", "Physiology or Medicine", "Peace",
                                  "Physics", "Economic Sciences"], columns="Year")
    sns.set_theme()
    f, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(x, annot=False, fmt="d", linewidths=0.05, ax=ax)
st.pyplot()

with st.echo(code_location='below'):
    quant = subj[["Category", "Total"]]
    colors = [plt.cm.Spectral(i / float(len(quant["Category"]))) for i in range(len(quant["Category"]))]
    plt.figure(figsize=(15, 8), dpi=80)
    squarify.plot(sizes=quant["Total"], label=quant["Category"], color=colors, alpha=0.8, value=quant["Total"])
    plt.title('Treemap of the number of the Nobel prizes per category')
    plt.axis('off')
st.pyplot()

with st.echo(code_location='below'):
    x = data_year[["Chemistry", "Literature", "Physiology or Medicine", "Peace", "Physics", "Economic Sciences"]].copy()
    x.index = data_year["Year"]
    x.plot.area(color=colors)
st.pyplot()


with st.echo(code_location='below'):
    strany = data_year[["Year", "male"]].copy()
    data["birth_countryNow"] = data["birth_countryNow"].fillna(data["org_founded_countryNow"])
    for i in set(data["birth_countryNow"].to_list()):
        strany[i] = 0
    strany = strany.drop(strany.columns[1], axis=1)
    for j in strany.columns.to_list()[1:-1]:
        for i in data_year["Year"]:
            strany.loc[i - 1901 - 3 * int(i / 1943)][j] = int(
                data[data["awardYear"] == int(i)]["birth_countryNow"].to_list().count(j))

    po_strane = pd.DataFrame(strany[strany.columns[1:]].sum().sort_values(ascending=False))

    po_strane = po_strane.iloc[0:25, :]
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')
    lowerLimit = 30
    labelPadding = 4
    max = int(po_strane.max())

    # Idea taken FROM https://www.python-graph-gallery.com/circular-barplot-basic
    slope = (max - lowerLimit) / max
    heights = slope * po_strane.iloc[:, 0] + lowerLimit
    width = 2 * np.pi / len(po_strane.index)
    indexes = list(range(1, len(po_strane.index) + 1))
    angles = [element * width for element in indexes]
    bars = ax.bar(angles, height=heights, width=width, bottom=lowerLimit, linewidth=2, edgecolor="white", color=colors)
    for bar, angle, height, label in zip(bars, angles, heights, po_strane.index.to_list()):
        rotation = np.rad2deg(angle)
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"
        ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding, s=str(label) + " " +
                                                                           str(int(
                                                                               po_strane[po_strane.index == label].iloc[
                                                                               :, 0])), ha=alignment, va='center',
                rotation=rotation, rotation_mode="anchor")
    # END FROM https://www.python-graph-gallery.com/circular-barplot-basic

st.pyplot()



with st.echo(code_location='below'):
    country = pd.DataFrame(strany[strany.columns[2:]].sum())
    country = country.reset_index(level=0, drop=False)

    # FROM https://melaniesoek0120.medium.com/data-visualization-how-to-plot-a-map-with-geopandas-in-python-73b10dcd4b4b
    # Аббревиатура страны по названию
    def alpha3code(column):
        CODE = []
        for country in column:
            try:
                code = pycountry.countries.get(name=country).alpha_3
                # .alpha_3 means 3-letter country code
                # .alpha_2 means 2-letter country code
                CODE.append(code)
            except:
                CODE.append('None')
        return CODE

    # END FROM https://melaniesoek0120.medium.com/data-visualization-how-to-plot-a-map-with-geopandas-in-python-73b10dcd4b4b

    # Плохой датасет, некоторое пришлось прописывать ручками
    country['CODE'] = alpha3code(country["index"])
    country.loc[country['index'] == "Czech Republic", 'CODE'] = "CZE"
    country.loc[country['index'] == "United Kingdom", 0] = 105
    country.loc[country['index'] == "Vietnam", 'CODE'] = "VNM"
    country.loc[country['index'] == "Iran", 'CODE'] = "IRN"
    country.loc[country['index'] == "Venezuela", 'CODE'] = "VEN"
    country.loc[country['index'] == "South Korea", 'CODE'] = "KOR"
    country.loc[country['index'] == "Russia", 'CODE'] = "RUS"
    country.loc[country['index'] == "the Netherlands", 'CODE'] = "NLD"
    country.loc[country['index'] == "USA", 'CODE'] = "USA"

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    country = country.drop(country[country["CODE"] == "None"].index)
    ppp = world.merge(country[["CODE", 0]], left_on='iso_a3', right_on='CODE')

    # Using the example (close to documentation) from https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746
    plt.figure(figsize=(9, 9))
    ax = ppp.dropna().plot(column=0, cmap="OrRd", figsize=(20, 20), scheme='quantiles', k=5, legend=True)
    ax.set_axis_off()
    ax.get_legend().set_bbox_to_anchor((.12, .12))
    ax.get_figure()
st.pyplot()

"Интересно, а о чём же вообще все эти научные работы? Наверняка у них очень заумные названия..." \
"Но должны же слова иногда повторяться?"
with st.echo(code_location='below'):
    # idea from https://plotly.com/python/choropleth-maps/
    fig = go.Figure(data=go.Choropleth(locations = country['CODE'],z = country[0],text = country['index'],colorscale = "spectral",
        autocolorscale=False,reversescale=False,
        marker_line_color='darkgray',marker_line_width=0.5,colorbar_title = 'No.'))

    fig.update_layout(
        title_text='Nobel prizes per country',geo=dict(showframe=False,showcoastlines=False,
            projection_type='equirectangular'),annotations = [dict(x=0.55, y=0.1, xref='paper',yref='paper', text="Можно туда сюда поводить потыкать",showarrow = False)])
st.write(fig)


with st.echo(code_location='below'):
    s = st.selectbox("Chose your fighter:", b)
    text = ""
    for i in data[data["category"]==s]["motivation"]:
        text+= " "+i
    mask = np.array(Image.open("Literture.png"))
    #FROM https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32
    mask_colors = ImageColorGenerator(mask)
    wc = WordCloud(stopwords=STOPWORDS, mask=mask, background_color="white", max_words=2000, max_font_size=256,
    random_state=42, width=mask.shape[1],height=mask.shape[0],color_func=mask_colors)
    #END FROM https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
st.pyplot()



with st.echo(code_location='below'):
    univ = data_year[["Year", "male"]].copy()

    for i in set(data["category"].to_list()):
        univ[i] = 0

    for j in ["Literature","Chemistry", "Physiology or Medicine", "Peace", "Physics", "Economic Sciences"]:
        for i in data_year["Year"]:
            if i == 1901:
                univ.loc[i - 1901 - 3 * int(i / 1943)][j] = int(
                    data[data["awardYear"] == int(i)]["category"].to_list().count(j))
            else:
                univ.loc[i - 1901 - 3 * int(i / 1943)][j] = int(
                    data[data["awardYear"] == int(i)]["category"].to_list().count(j))+int(univ.loc[i - 1902 - 3 * int(i / 1943)][j])
    univ = univ.set_index("Year").drop(columns="male")
    print(univ)
    print(univ.sort_values(univ.loc["2019"],axis=1))



the_plot = st.pyplot(plt)
def animate(i, x, y, colors):
    ax.barh(x, width=y, color=colors)
    ax.set_title(i,)
    the_plot.pyplot(plt)

fig, ax = plt.subplots()
for i in range(110):
    width = univ.iloc[i].values
    animate(i, univ.iloc[i].index, width, colors)
    time.sleep(0.2)



aue  = st.checkbox("Го еще раз?",["No","Yes"])



"""fig, ax = plt.subplots()
camera = Camera(fig)
for i in range(20):
    ax.barh(univ.iloc[i].index, width=univ.iloc[i].values, color=colors)
    camera.snap()
camera.animate()
st.pyplot()"""



yes  = st.checkbox("Хочу посмотреть сразу на все предметы")
if yes == True:
    with st.echo(code_location='below'):
        for j in b:
            text = ""
            for i in data[data["category"] == j]["motivation"]:
                text += " " + i
            wordcloud = WordCloud(width=400, height=400, margin=0, background_color="white").generate(text)
            plt.subplot(2, 3, b.index(j) + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(str(j))
            plt.margins(x=0, y=0)


if page == "Search":
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)


    data = pd.read_csv('complete.csv', delimiter=',').sort_values("awardYear").reset_index()
    del data["index"]

    # Повтор из прошлого цикла
    data["gender"] = data["gender"].fillna("Organization")
    data_year = pd.DataFrame({"Year": range(1901, 1940), "male": 0, "female": 0, "Organization": 0})
    data_year2 = pd.DataFrame({"Year": range(1943, 2020), "male": 0, "female": 0, "Organization": 0})
    data_year = pd.concat([data_year, data_year2], ignore_index=True)
    for j in ["male", "female", "Organization"]:
        for i in range(1901, 1939):
            data_year.loc[i - 1901][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))
        for i in range(1943, 2020):
            data_year.loc[i - 1904][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))
    a = ["Chemistry", "Literature", "Physiology or Medicine", "Peace", "Physics", "Economic Sciences"]

    cat = st.selectbox('Выберите интересующую вас область:', a)
    year = st.selectbox('Выберите интересующий вас год:', range(1901,2020))
    if year>1939 and year < 1943:
        "В этом году нобелевскую премию по данному предмету никто не получал. Да и по другим претметам тоже. " \
        "Война всё-так дело серьезное"

    else:
        if len(data[data["awardYear"] == year][data["category"] == cat]) ==0:
            "В этом году по"+" "+cat+" никто не получал премию. Знаете почему? Её тогда ещё не было)) " \
                                     "Она появилась в 1969."

        elif len(data[data["awardYear"] == year][data["category"] == cat]) ==1:
            "В этом году Нобелевскую премию по "+str(cat)+" получил "+ \
            data[data["awardYear"] == year][data["category"] == cat]["name"].iloc[0]
            "На момент"




"Интересно, а о чём же вообще все эти научные работы? Наверняка у них очень заумные названия..." \
"Но должны же слова иногда повторяться?"

yes  = st.checkbox("Хочу посмотреть сразу на все предметы")
if yes == True:
    with st.echo(code_location='below'):
        for j in b:
            text = ""
            for i in data[data["category"] == j]["motivation"]:
                text += " " + i
            wordcloud = WordCloud(width=400, height=400, margin=0, background_color="white").generate(text)
            plt.subplot(2, 3, b.index(j) + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(str(j))
            plt.margins(x=0, y=0)
st.pyplot()



if page == "Search":
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)


    data = pd.read_csv('complete.csv', delimiter=',').sort_values("awardYear").reset_index()
    del data["index"]

    # Повтор из прошлого цикла
    data["gender"] = data["gender"].fillna("Organization")
    data_year = pd.DataFrame({"Year": range(1901, 1940), "male": 0, "female": 0, "Organization": 0})
    data_year2 = pd.DataFrame({"Year": range(1943, 2020), "male": 0, "female": 0, "Organization": 0})
    data_year = pd.concat([data_year, data_year2], ignore_index=True)
    for j in ["male", "female", "Organization"]:
        for i in range(1901, 1939):
            data_year.loc[i - 1901][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))
        for i in range(1943, 2020):
            data_year.loc[i - 1904][j] = int(data[data["awardYear"] == int(i)]["gender"].to_list().count(j))
    a = ["Chemistry", "Literature", "Physiology or Medicine", "Peace", "Physics", "Economic Sciences"]

    cat = st.selectbox('Выберите интересующую вас область:', a)
    year = st.selectbox('Выберите интересующий вас год:', range(1901,2020))
    if year>1939 and year < 1943:
        "В этом году нобелевскую премию по данному предмету никто не получал. Да и по другим претметам тоже. " \
        "Война всё-так дело серьезное"

    else:
        if len(data[data["awardYear"] == year][data["category"] == cat]) ==0:
            "В этом году по"+" "+cat+" никто не получал премию. Знаете почему? Её тогда ещё не было)) " \
                                     "Она появилась в 1969."

        elif len(data[data["awardYear"] == year][data["category"] == cat]) ==1:
            "В этом году Нобелевскую премию по "+str(cat)+" получил "+ \
            data[data["awardYear"] == year][data["category"] == cat]["name"].iloc[0]
            "На момент"

