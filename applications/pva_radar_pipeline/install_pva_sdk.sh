#!/bin/bash
set -x  # Enable debug mode to print each command
# Add verbose output
echo "Starting PVA SDK installation script"
echo "CUPVA_VERSIONS=$CUPVA_VERSIONS"

IFS=',' read -r -a versions <<< "$CUPVA_VERSIONS"
echo "Parsed versions: ${versions[@]}"
for version in "${versions[@]}"; do
    echo "Installing CUPVA version $version..."
    PVA_SDK_MAJOR_MINOR_VERSION=$(echo ${version} | sed -E 's/([0-9]+\.[0-9]+)\..*/\1/' | sed 's/~.*//')

    # Function to check if a URL exists
    check_url() {
        local url="$1"
        if curl --head --silent --fail "$url" > /dev/null; then
            return 0  # URL exists
        else
            return 1  # URL does not exist
        fi
    }

    # Set base URL by checking where the version exists
    RELEASES_URL="https://urm.nvidia.com/artifactory/sw-cupva-sdk-generic-local/releases/${version}"
    RC_URL="https://urm.nvidia.com/artifactory/sw-cupva-sdk-generic-local/release_candidates/${version}"

    if check_url "${RELEASES_URL}/"; then
        BASE_URL="$RELEASES_URL"
        echo "Found version $version in releases"
    elif check_url "${RC_URL}/"; then
        BASE_URL="$RC_URL"
        echo "Found version $version in release_candidates"
    else
        echo "ERROR: Version $version not found in either releases or release_candidates"
        continue  # Skip this version and try the next one
    fi

    # Download packages from the selected BASE_URL
    wget -q ${BASE_URL}/local_installers/public/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-core_${version}_all.deb
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download core package"
        continue
    fi

    wget -q ${BASE_URL}/local_installers/private/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-internal_${version}_all.deb
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download internal package"
        continue
    fi

    wget -q ${BASE_URL}/local_installers/private/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-qa_${version}_all.deb
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download qa package"
        continue
    fi

    wget -q ${BASE_URL}/local_installers/synopsys_licensed/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-gen2_${version}_all.deb
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download gen2 package"
        continue
    fi

    apt-get install -y \
        ./pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-core_${version}_all.deb \
        ./pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-internal_${version}_all.deb \
        ./pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-qa_${version}_all.deb \
        ./pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-gen2_${version}_all.deb
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install core packages"
        continue
    fi

    # Check if version is >= 2.6.0 for GEN3 packages
    install_gen3=$( [ "$(printf '%s\n' "2.6.0" "$version" | sort -V | head -n1)" == "2.6.0" ] && echo true || echo false )
    # Download GEN3 package if version >= 2.6.0
    if [ "$install_gen3" = true ]; then
        wget -q ${BASE_URL}/local_installers/synopsys_licensed/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-gen3_${version}_all.deb
        if [ $? -eq 0 ]; then
            apt-get install -y ./pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-gen3_${version}_all.deb
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to install gen3 package"
            fi
        else
            echo "ERROR: Failed to download gen3 package"
        fi
    fi

    # Check if version is >= 2.7.0 for native-pvaumd and sim-pvaumd packages
    install_pvaumd_packages=$( [ "$(printf '%s\n' "2.7.0" "$version" | sort -V | head -n1)" == "2.7.0" ] && echo true || echo false )

    # Update apt repository cache once for all repository package installations in this iteration
    apt-get update

    # Install core development packages
    apt-get install -y \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-l4t-dev \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-l4t-cross \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-dev \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-gen2-dev \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-debug-tools \
        pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-l4t-tegralibs
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install some dev packages"
    fi

    # Install GEN3 packages for SDK version 2.6.0 and later
    if [ "$install_gen3" = true ]; then
        apt-get install -y \
            pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-gen3-dev \
            pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-gen3-ppe-dev \
            pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-ppe-dev
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install gen3 dev packages"
        fi
    fi

    # Install native-pvaumd packages for SDK version 2.7.0 and later
    if [ "$install_pvaumd_packages" = true ]; then
        apt-get install -y \
            pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-pvaumd
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install pvaumd packages"
        fi
    fi
done

# Clean up
rm -f *.deb
rm -rf /var/lib/apt/lists/*

# List installed PVA SDK packages to verify
echo "Installed PVA SDK packages:"
dpkg -l | grep pva-sdk