library(readr)
library(ggplot2)
library(scales)

# ============================================================
# DATA
# ============================================================

tier_df <- read_csv(
  "data/output_metrics/tier_area.csv",
  show_col_types = FALSE
)

# Pastikan tier numerik dan urut
tier_df$tier <- as.integer(tier_df$tier)

tier_df <- tier_df[
  order(tier_df$tier),
]

# Total ROI dihitung dari data
total_roi <- sum(tier_df$luas_ha, na.rm = TRUE)

# Urutan kategori: Tier 0 -> Tier 4
tier_df$kategori <- factor(
  tier_df$kategori,
  levels = rev(tier_df$kategori)
)

# Label kategori yang lebih ringkas untuk sumbu
tier_df$label <- paste0(
  "Tier ",
  tier_df$tier,
  "  |  ",
  tier_df$kategori
)

tier_df$label <- factor(
  tier_df$label,
  levels = rev(tier_df$label)
)

# ============================================================
# FIGURE
# ============================================================

plot <- ggplot(
  tier_df,
  aes(
    x = luas_ha,
    y = label
  )
) +

  # ----------------------------------------------------------
  # Main bars
  # ----------------------------------------------------------

  geom_col(
    width = 0.62,
    fill = "#3B6E8F"
  ) +

  # ----------------------------------------------------------
  # Percentage labels inside bars
  # ----------------------------------------------------------

  geom_text(
    aes(
      label = paste0(
        number(
          persen,
          accuracy = 0.01,
          decimal.mark = "."
        ),
        "%"
      )
    ),
    hjust = 1.12,
    colour = "white",
    fontface = "bold",
    size = 3.5
  ) +

  # ----------------------------------------------------------
  # Area labels outside bars
  # ----------------------------------------------------------

  geom_text(
    aes(
      x = luas_ha,
      label = paste0(
        number(
          luas_ha,
          accuracy = 0.01,
          big.mark = ",",
          decimal.mark = "."
        ),
        " ha"
      )
    ),
    hjust = -0.10,
    size = 3.4,
    colour = "black"
  ) +

  # ----------------------------------------------------------
  # X axis
  # ----------------------------------------------------------

  scale_x_continuous(
    labels = label_number(
      big.mark = ",",
      decimal.mark = "."
    ),
    expand = expansion(
      mult = c(0, 0.17)
    )
  ) +

  # ----------------------------------------------------------
  # Labels
  # ----------------------------------------------------------

  labs(
    title = "Distirbusi Area berdasarkan Tier Bukti Kausal",
    subtitle = paste0(
      "Total area studi = ",
      number(
        total_roi,
        accuracy = 0.01,
        big.mark = ",",
        decimal.mark = "."
      ),
      " ha"
    ),
    x = "Area (ha)",
    y = NULL,
    caption = "Percentages represent the proportion of the total study area."
  ) +

  # ----------------------------------------------------------
  # Theme
  # ----------------------------------------------------------

  theme_classic(
    base_size = 11
  ) +
  theme(
    # Title
    plot.title = element_text(
      size = 15,
      face = "bold",
      colour = "black",
      margin = margin(
        b = 4
      )
    ),

    # Subtitle
    plot.subtitle = element_text(
      size = 10.5,
      colour = "grey30",
      margin = margin(
        b = 14
      )
    ),

    # X axis title
    axis.title.x = element_text(
      size = 10.5,
      face = "bold",
      margin = margin(
        t = 9
      )
    ),

    # X axis
    axis.text.x = element_text(
      size = 9,
      colour = "black"
    ),

    # Y axis
    axis.text.y = element_text(
      size = 9.5,
      colour = "black",
      lineheight = 1.0
    ),
    axis.ticks.y = element_blank(),

    # Grid
    panel.grid.major.x = element_line(
      colour = "grey88",
      linewidth = 0.3
    ),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),

    # Caption
    plot.caption = element_text(
      size = 8.5,
      colour = "grey40",
      hjust = 0,
      margin = margin(
        t = 10
      )
    ),

    # Margins
    plot.margin = margin(
      t = 12,
      r = 55,
      b = 12,
      l = 12
    )
  ) +

  # Allow labels outside plotting region
  coord_cartesian(
    clip = "off"
  )

print(plot)

# ============================================================
# EXPORT
# ============================================================

dir.create(
  "results/figures",
  recursive = TRUE,
  showWarnings = FALSE
)

# Vector format - preferred for manuscript
ggsave(
  "results/figures/causal_evidence_tier_distribution.pdf",
  plot = plot,
  width = 180,
  height = 115,
  units = "mm",
  device = cairo_pdf
)

# High-resolution raster - useful for submission systems
ggsave(
  "results/figures/causal_evidence_tier_distribution.png",
  plot = plot,
  width = 180,
  height = 115,
  units = "mm",
  dpi = 600,
  bg = "white"
)
