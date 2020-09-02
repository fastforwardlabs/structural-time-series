library(shiny)
library(tidyverse)


ui <- fluidPage(
    titlePanel("California Electricity Demand"),
    fluidRow(
        column(12,
        p("A minimal explorer app for California Electricity Demand data over the past ~5 years."),
        p("The top chart is brushable: click and drag to select a region.
           The lower chart will show a zoomed view of the selected time period."),
        p("The charts may take a moment to load while data is read."))
    ),
    fluidRow(column(12, h3("Date range selector - brush me!"))),
    fluidRow(
        column(12,
            plotOutput("controlPlot", height = 300,
                dblclick = "plot1_dblclick",
                brush = brushOpts(
                    id = "brush",
                    direction = "x",
                    resetOnNew = TRUE,
                    stroke = "#00828c",
                    fill = "#00828c"
                )
            )
        )
    ),
    fluidRow(column(12, h3("Zoomed view"))),
    fluidRow(
        column(12,
            plotOutput("zoomPlot", height = 300)
        )
    )
)


server <- function(input, output) {

    demand <- read_csv("data/demand.csv") %>%
        mutate(y = y / 1000) # transform from megawatt-hours to gigawatt-hours

    range <- reactiveValues(x = NULL)

    output$controlPlot <- renderPlot({
        ggplot(demand, aes(x=ds, y=y)) +
            geom_line(color="#aaaaaa") +
            labs(x = "Datetime (hourly)", y = "Demand (gigawatt-hours)") +
            theme_light()
    })

    output$zoomPlot <- renderPlot({
        ggplot(demand, aes(x=ds, y=y)) +
            geom_line(color="#00828c") +
            coord_cartesian(xlim = range$x) +
            labs(x = "Datetime (hourly)", y = "Demand (gigawatt-hours)") +
            scale_x_datetime(breaks = scales::breaks_pretty(12)) +
            theme_light()
    })

    observe({
        brush <- input$brush
        if (!is.null(brush)) {
            range$x <- as.POSIXct(c(brush$xmin, brush$xmax), origin="1970-01-01")
        } else {
            range$x <- NULL
        }
    })
}

app <- shinyApp(ui, server)

runApp(
    app,
    port = as.numeric(Sys.getenv("CDSW_READONLY_PORT")),
    host = "127.0.0.1",
    launch.browser = "FALSE"
)