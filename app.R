library(shiny)
library(tidyverse)


ui <- fluidPage(
    fluidRow(
        plotOutput("controlPlot", height = 300,
            dblclick = "plot1_dblclick",
            brush = brushOpts(
                id = "brush",
                direction = "x",
                resetOnNew = TRUE
            )
        )
    ),
    fluidRow(
        plotOutput("zoomPlot", height = 300)
    )
)


server <- function(input, output) {

    demand <- read_csv('demand.csv')

    range <- reactiveValues(x = NULL)
    
    output$controlPlot <- renderPlot({
        ggplot(demand, aes(x=datetime, y=demand)) +
            geom_line() +
            theme_minimal()
    })

    output$zoomPlot <- renderPlot({
        ggplot(demand, aes(x=datetime, y=demand)) +
            geom_line() +
            coord_cartesian(xlim = range$x) +
            theme_minimal()
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
