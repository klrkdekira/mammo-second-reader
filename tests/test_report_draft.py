from src.reporting.report_draft import create_report_draft


def test_create_report_draft_preserves_template_and_relative_links(tmp_path):
    template = tmp_path / "documents/ReportTemplate.md"
    output = tmp_path / "documents/FinalReport.md"
    template.parent.mkdir()
    template.write_text(
        "# Report\n\n![Plot](../results/figures/plot.png)\n"
        "![Design](assets/design.png)\n"
    )

    assert create_report_draft(template, output)
    assert "](../results/figures/plot.png)" in template.read_text()
    assert "](assets/design.png)" in template.read_text()
    assert "](../results/figures/plot.png)" in output.read_text()
    assert "](assets/design.png)" in output.read_text()


def test_create_report_draft_does_not_overwrite_existing_work(tmp_path):
    template = tmp_path / "template.md"
    output = tmp_path / "FinalReport.md"
    template.write_text("template\n")
    output.write_text("edited\n")

    assert not create_report_draft(template, output)
    assert output.read_text() == "edited\n"

    assert create_report_draft(template, output, force=True)
    assert output.read_text() == "template\n"
