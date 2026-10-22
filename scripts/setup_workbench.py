"""Create only this project's FDR login secret. No credentials are printed or written to Git."""

import secrets

import modal


def main():
    modal.Dict.objects.create("k2-interp-live", environment_name="FDR", allow_existing=True)
    names = {item.name for item in modal.Secret.objects.list(environment_name="FDR")}
    if "k2-interp-workbench-auth" in names:
        print("Workbench auth secret already exists in FDR; preserved.")
        return
    modal.Secret.objects.create(
        "k2-interp-workbench-auth",
        {"K2_WORKBENCH_SIGNING_KEY": secrets.token_urlsafe(48)},
        environment_name="FDR",
    )
    print("Created k2-interp-workbench-auth in FDR.")


if __name__ == "__main__":
    main()
